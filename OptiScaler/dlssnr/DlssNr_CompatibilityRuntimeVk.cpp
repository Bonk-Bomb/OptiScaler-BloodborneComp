#include "pch.h"
#include "DlssNr_CompatibilityRuntimeVk.h"
#include "DlssNr_RuntimeImports.h"

#include <Config.h>
#include <Logger.h>
#include <State.h>
#include <Util.h>

#include <psapi.h>
#include <algorithm>
#include <atomic>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#pragma comment(lib, "psapi.lib")

// Native Vulkan twin of DlssNr_CompatibilityRuntime.cpp. The caller-path adapter below is the same
// mechanism, kept in this translation unit so the D3D12 backend is untouched.

namespace DlssNr
{
namespace
{
std::recursive_mutex registryMutex;

// Only the model's import is redirected. Outside a direct NR call, even that import reports the real path.
thread_local HMODULE callerAlias = nullptr;

// Preserve whatever the runtime's import pointed at before us (the system, a loader, an overlay, or the
// D3D12 backend's own adapter) for every ordinary path query.
std::atomic<decltype(&GetModuleFileNameW)> originalPathW { &GetModuleFileNameW };
std::atomic<decltype(&GetModuleFileNameA)> originalPathA { &GetModuleFileNameA };

DWORD WINAPI CallerPathVkW(HMODULE queried, LPWSTR path, DWORD capacity)
{
    const auto original = originalPathW.load();
    if (callerAlias && queried == callerAlias)
    {
        constexpr wchar_t alias[] = L"nvngx.dll";
        if (!capacity)
        {
            SetLastError(ERROR_INSUFFICIENT_BUFFER);
            return 0;
        }
        const DWORD copied = (DWORD) std::min<size_t>(capacity - 1, std::size(alias) - 1);
        memcpy(path, alias, copied * sizeof(wchar_t));
        path[copied] = 0;
        if (capacity < std::size(alias))
        {
            SetLastError(ERROR_INSUFFICIENT_BUFFER);
            return capacity;
        }
        return copied;
    }
    return original(queried, path, capacity);
}

DWORD WINAPI CallerPathVkA(HMODULE queried, LPSTR path, DWORD capacity)
{
    const auto original = originalPathA.load();
    if (callerAlias && queried == callerAlias)
    {
        constexpr char alias[] = "nvngx.dll";
        if (!capacity)
        {
            SetLastError(ERROR_INSUFFICIENT_BUFFER);
            return 0;
        }
        const DWORD copied = (DWORD) std::min<size_t>(capacity - 1, std::size(alias) - 1);
        memcpy(path, alias, copied);
        path[copied] = 0;
        if (capacity < std::size(alias))
        {
            SetLastError(ERROR_INSUFFICIENT_BUFFER);
            return capacity;
        }
        return copied;
    }
    return original(queried, path, capacity);
}

// Marks this OptiScaler module as the caller for the duration of one direct call.
struct CallerScope
{
    HMODULE previous = callerAlias;
    CallerScope()
    {
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&CallerPathVkW), &callerAlias);
    }
    ~CallerScope() { callerAlias = previous; }
};

bool ReplaceImport(void** slot, void* expected, void* replacement)
{
    if (*slot != expected)
    {
        LOG_ERROR("NR compatibility (Vulkan): import changed before replacement: slot={} expected={} actual={}",
                  (void*) slot, expected, *slot);
        return false;
    }
    DWORD protection = 0;
    if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &protection))
    {
        LOG_ERROR("NR compatibility (Vulkan): import protection change failed: slot={} error={}", (void*) slot,
                  GetLastError());
        return false;
    }
    const bool replaced = InterlockedCompareExchangePointer(slot, replacement, expected) == expected;
    DWORD ignored = 0;
    if (!VirtualProtect(slot, sizeof(void*), protection, &ignored))
        LOG_WARN("NR compatibility (Vulkan): import protection restore failed: slot={} error={}", (void*) slot,
                 GetLastError());
    if (!replaced)
        LOG_ERROR("NR compatibility (Vulkan): import changed concurrently: slot={}", (void*) slot);
    return replaced;
}
} // namespace

struct CompatibilityRuntimeVk::Module
{
    struct Binding
    {
        RuntimeImports::Slot slot;
        void* original;
    };

    HMODULE handle = nullptr;
    bool borrowed = false;
    std::vector<Binding> imports;
    std::filesystem::path path;
    std::recursive_mutex mutex;
    std::set<VkDevice> initializedDevices;

    // The runtime's own Vulkan surface. Init_Ext's sixth argument is passed null, which is what the 0.2.0
    // forwarder did on native Vulkan games (Bloodborne on shadPS4 among them) with this same runtime.
    using Init = NVSDK_NGX_Result (*)(unsigned long long, const wchar_t*, VkInstance, VkPhysicalDevice, VkDevice,
                                      const void*, unsigned int);
    using Create = NVSDK_NGX_Result (*)(VkCommandBuffer, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
    using Evaluate = NVSDK_NGX_Result (*)(VkCommandBuffer, const NVSDK_NGX_Handle*, NVSDK_NGX_Parameter*, void*);
    using Release = NVSDK_NGX_Result (*)(NVSDK_NGX_Handle*);
    using Shutdown = NVSDK_NGX_Result (*)(VkDevice);

    Init init = nullptr;
    Create create = nullptr;
    Evaluate evaluate = nullptr;
    Release release = nullptr;
    Shutdown shutdown = nullptr; // resolved for diagnostics only; see ~CompatibilityRuntimeVk

    ~Module()
    {
        std::lock_guard registryLock(registryMutex);
        for (const auto& binding : imports)
            ReplaceImport(binding.slot.address,
                          binding.slot.wide ? reinterpret_cast<void*>(&CallerPathVkW)
                                            : reinterpret_cast<void*>(&CallerPathVkA),
                          binding.original);
        // The runtime stays mapped until process exit, as it did under the 0.2.0 forwarder. Unloading a
        // model that may still own device-wide NGX state during teardown is not worth the risk.
    }
};

std::shared_ptr<CompatibilityRuntimeVk> CompatibilityRuntimeVk::Open(const std::filesystem::path& candidate,
                                                                     VkInstance instance,
                                                                     VkPhysicalDevice physicalDevice, VkDevice device,
                                                                     const std::filesystem::path& dataPath)
{
    if (instance == VK_NULL_HANDLE || physicalDevice == VK_NULL_HANDLE || device == VK_NULL_HANDLE)
        return {};

    static std::weak_ptr<Module> liveModule;
    static std::map<VkDevice, std::weak_ptr<CompatibilityRuntimeVk>> devices;
    std::lock_guard lock(registryMutex);

    try
    {
        std::error_code ec;
        if (!std::filesystem::exists(candidate, ec))
            return {};

        auto path = std::filesystem::absolute(candidate).lexically_normal();
        auto loaded = liveModule.lock();
        if (loaded && !std::filesystem::equivalent(loaded->path, path, ec))
            return {};

        for (auto it = devices.begin(); it != devices.end();)
            if (it->second.expired())
                it = devices.erase(it);
            else
                ++it;
        if (auto it = devices.find(device); it != devices.end())
            if (auto existing = it->second.lock())
                return existing;

        if (!loaded)
        {
            loaded = std::make_shared<Module>();
            loaded->path = path;

            // Take our own reference if NGX or another loader already mapped this exact runtime.
            loaded->borrowed = GetModuleHandleExW(0, path.c_str(), &loaded->handle) != FALSE;
            if (!loaded->handle)
                loaded->handle = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (!loaded->handle)
            {
                LOG_ERROR("NR compatibility (Vulkan): cannot load {} ({})", path.string(), GetLastError());
                return {};
            }

            auto symbol = [&](const char* name) { return GetProcAddress(loaded->handle, name); };
            loaded->init = reinterpret_cast<Module::Init>(symbol("NVSDK_NGX_VULKAN_Init_Ext"));
            loaded->create = reinterpret_cast<Module::Create>(symbol("NVSDK_NGX_VULKAN_CreateFeature"));
            loaded->evaluate = reinterpret_cast<Module::Evaluate>(symbol("NVSDK_NGX_VULKAN_EvaluateFeature"));
            loaded->release = reinterpret_cast<Module::Release>(symbol("NVSDK_NGX_VULKAN_ReleaseFeature"));
            loaded->shutdown = reinterpret_cast<Module::Shutdown>(symbol("NVSDK_NGX_VULKAN_Shutdown1"));
            if (!loaded->init || !loaded->create || !loaded->evaluate || !loaded->release)
            {
                LOG_ERROR("NR compatibility (Vulkan): {} is missing required Vulkan NR exports (init={} create={} "
                          "evaluate={} release={})",
                          path.string(), loaded->init != nullptr, loaded->create != nullptr,
                          loaded->evaluate != nullptr, loaded->release != nullptr);
                return {};
            }

            MODULEINFO image {};
            std::vector<RuntimeImports::Slot> slots;
            if (!GetModuleInformation(GetCurrentProcess(), loaded->handle, &image, sizeof(image)) ||
                !RuntimeImports::Find({ static_cast<unsigned char*>(image.lpBaseOfDll), image.SizeOfImage }, slots))
            {
                LOG_ERROR("NR compatibility (Vulkan): invalid runtime import table in {}", path.string());
                return {};
            }

            // Resolve named imports from this build; never a version-specific address.
            void* previousW = nullptr;
            void* previousA = nullptr;
            for (const auto& slot : slots)
            {
                auto*& previous = slot.wide ? previousW : previousA;
                if (!*slot.address || (previous && previous != *slot.address))
                {
                    LOG_ERROR("NR compatibility (Vulkan): conflicting caller-path import targets in {}", path.string());
                    return {};
                }
                previous = *slot.address;
            }
            if (previousW)
                originalPathW = reinterpret_cast<decltype(&GetModuleFileNameW)>(previousW);
            if (previousA)
                originalPathA = reinterpret_cast<decltype(&GetModuleFileNameA)>(previousA);

            loaded->imports.reserve(slots.size());
            for (const auto& slot : slots)
            {
                auto* previous = slot.wide ? previousW : previousA;
                if (!ReplaceImport(slot.address, previous,
                                   slot.wide ? reinterpret_cast<void*>(&CallerPathVkW)
                                             : reinterpret_cast<void*>(&CallerPathVkA)))
                {
                    LOG_ERROR("NR compatibility (Vulkan): cannot adapt caller-path import in {}", path.string());
                    return {};
                }
                loaded->imports.push_back({ slot, previous });
            }

            liveModule = loaded;
            LOG_INFO("NR compatibility (Vulkan): {} {} with {} named caller-path imports, no helper DLL",
                     loaded->borrowed ? "retained existing runtime" : "loaded", path.string(), slots.size());
        }

        auto owner = std::shared_ptr<CompatibilityRuntimeVk>(new CompatibilityRuntimeVk());
        owner->module = loaded;
        owner->device = device;

        std::lock_guard runtimeLock(loaded->mutex);
        if (!loaded->initializedDevices.contains(device))
        {
            CallerScope caller;
            const auto writablePath = dataPath.empty() ? std::filesystem::temp_directory_path() : dataPath;
            const auto result =
                loaded->init(0x0ull, writablePath.c_str(), instance, physicalDevice, device, nullptr, 0x15u);
            LOG_INFO("NR compatibility (Vulkan): Init_Ext result=0x{:08X}", (unsigned) result);
            if (result != NVSDK_NGX_Result_Success)
                return {};
            loaded->initializedDevices.insert(device);
        }

        owner->initialized = true;
        devices[device] = owner;
        return owner;
    }
    catch (const std::exception& error)
    {
        LOG_ERROR("NR compatibility (Vulkan): {}", error.what());
        return {};
    }
}

std::shared_ptr<CompatibilityRuntimeVk>
CompatibilityRuntimeVk::TryOpen(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device)
{
    std::vector<std::filesystem::path> paths;
    for (const auto& path : State::Instance().NVNGX_FeatureInfo_Paths)
        paths.emplace_back(path);
    paths.push_back(Util::ExePath().parent_path());
    paths.push_back(Util::DllPath().parent_path());
    if (Config::Instance()->MainDllPath.has_value())
        paths.emplace_back(Config::Instance()->MainDllPath.value());

    std::set<std::wstring> visited;
    for (const auto& path : paths)
    {
        std::error_code ec;
        const auto absolute = std::filesystem::absolute(path, ec);
        if (ec || !visited.insert(Util::ToLower(absolute.lexically_normal().wstring())).second)
            continue;
        if (auto runtime = Open(path / L"nvngx_dlssnr.dll", instance, physicalDevice, device,
                                State::Instance().NVNGX_ApplicationDataPath))
            return runtime;
    }

    LOG_INFO("NR compatibility (Vulkan): no usable nvngx_dlssnr.dll found; preserving driver failure");
    return {};
}

CompatibilityRuntimeVk::~CompatibilityRuntimeVk()
{
    if (!module)
        return;

    std::lock_guard registryLock(registryMutex);
    std::lock_guard lock(module->mutex);

    // Features are released by their owner before this runs. Device-wide NGX state is left in place:
    // Shutdown1 is not called because this destructor can run during device or process teardown, and the
    // 0.2.0 forwarder (which this route mirrors) never called it either.
    if (initialized)
        module->initializedDevices.erase(device);
}

NVSDK_NGX_Result CompatibilityRuntimeVk::Create(VkCommandBuffer commands, NVSDK_NGX_Parameter* params,
                                                NVSDK_NGX_Handle** feature)
{
    std::lock_guard lock(module->mutex);
    CallerScope caller;
    // Assigned, not returned: keep this frame on the stack so the model's caller is this module.
    volatile auto result = module->create(commands, static_cast<NVSDK_NGX_Feature>(18), params, feature);
    return result;
}

NVSDK_NGX_Result CompatibilityRuntimeVk::Evaluate(VkCommandBuffer commands, const NVSDK_NGX_Handle* feature,
                                                  NVSDK_NGX_Parameter* params)
{
    std::lock_guard lock(module->mutex);
    CallerScope caller;
    volatile auto result = module->evaluate(commands, feature, params, nullptr);
    return result;
}

NVSDK_NGX_Result CompatibilityRuntimeVk::Release(NVSDK_NGX_Handle* feature)
{
    std::lock_guard lock(module->mutex);
    CallerScope caller;
    volatile auto result = module->release(feature);
    return result;
}
} // namespace DlssNr
