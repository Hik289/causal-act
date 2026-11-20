/* Enumerate Vulkan device UUIDs without creating a device or allocating GPU memory. */
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

int main(void) {
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "CausalAct device identity", .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo create = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app };
    VkInstance instance;
    VkResult result = vkCreateInstance(&create, NULL, &instance);
    if (result != VK_SUCCESS) { fprintf(stderr, "vkCreateInstance: %d\n", result); return 1; }
    uint32_t count = 0;
    result = vkEnumeratePhysicalDevices(instance, &count, NULL);
    if (result != VK_SUCCESS || !count) { vkDestroyInstance(instance, NULL); return 2; }
    VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
    if (!devices) { vkDestroyInstance(instance, NULL); return 3; }
    result = vkEnumeratePhysicalDevices(instance, &count, devices);
    if (result != VK_SUCCESS) { free(devices); vkDestroyInstance(instance, NULL); return 4; }
    printf("[");
    for (uint32_t index = 0; index < count; ++index) {
        VkPhysicalDeviceIDProperties id = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
        VkPhysicalDeviceProperties2 properties = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &id };
        vkGetPhysicalDeviceProperties2(devices[index], &properties);
        printf("%s{\"vulkan_index\":%u,\"vendor_id\":%u,\"device_uuid\":\"",
            index ? "," : "", index, properties.properties.vendorID);
        for (uint32_t n = 0; n < VK_UUID_SIZE; ++n) printf("%02x", id.deviceUUID[n]);
        printf("\"}");
    }
    printf("]\n");
    free(devices);
    vkDestroyInstance(instance, NULL);
    return 0;
}
