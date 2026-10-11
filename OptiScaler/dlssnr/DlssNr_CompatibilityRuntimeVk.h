#pragma once

#include <vulkan/vulkan.h>
#include <nvsdk_ngx_vk.h>
#include <filesystem>
#include <memory>

namespace DlssNr
{
// Vulkan counterpart of CompatibilityRuntime.
//
// The RTX 20/30/40 compatibility build of nvngx_dlssnr.dll is refused by the NGX driver's signed loader,
// so the driver dispatcher answers CreateFeature(18) with 0xBAD0000B on those setups. The D3D12 path
// already recovers by loading the runtime directly; native Vulkan had no such route and stopped there.
//
// This loads the same runtime directly through its Vulkan exports (the ones the 0.2.0 forwarder proved on
// native Vulkan games) and answers the model's caller-path check exactly like the D3D12 backend does:
// only the runtime's own GetModuleFileName imports are redirected, only for the calling OptiScaler module,
// and only for the duration of a direct call. No helper DLL, no driver patch, no change on disk.
//
// Model calls only. Parameter maps stay driver-owned, as on the D3D12 path.
class CompatibilityRuntimeVk
{
  public:
    static std::shared_ptr<CompatibilityRuntimeVk> TryOpen(VkInstance instance, VkPhysicalDevice physicalDevice,
                                                           VkDevice device);
    static std::shared_ptr<CompatibilityRuntimeVk> Open(const std::filesystem::path& path, VkInstance instance,
                                                        VkPhysicalDevice physicalDevice, VkDevice device,
                                                        const std::filesystem::path& dataPath = {});
    ~CompatibilityRuntimeVk();
    CompatibilityRuntimeVk(const CompatibilityRuntimeVk&) = delete;
    CompatibilityRuntimeVk& operator=(const CompatibilityRuntimeVk&) = delete;

    NVSDK_NGX_Result Create(VkCommandBuffer commands, NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** feature);
    NVSDK_NGX_Result Evaluate(VkCommandBuffer commands, const NVSDK_NGX_Handle* feature, NVSDK_NGX_Parameter* params);
    NVSDK_NGX_Result Release(NVSDK_NGX_Handle* feature);

  private:
    struct Module;
    std::shared_ptr<Module> module;
    VkDevice device = VK_NULL_HANDLE;
    bool initialized = false;
    CompatibilityRuntimeVk() = default;
};
} // namespace DlssNr
