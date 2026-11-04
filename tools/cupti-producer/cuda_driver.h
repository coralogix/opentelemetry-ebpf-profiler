// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0
//
// Driver lookups of the producer: driver functions and GPU UUIDs, found in the
// CUDA driver the process already loaded.
#pragma once

#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <cuda.h>

/* ─── Driver functions ────────────────────────────────────────────────────────
 * The few driver functions the producer calls are looked up in the driver the
 * process already loaded, so the library does not link it. The driver may be
 * loaded without its symbols being visible to other libraries. */
static void *driver_function(const char *name)
{
  static void *g_driver;
  void *driver = __atomic_load_n(&g_driver, __ATOMIC_ACQUIRE);
  if (!driver) {
    driver = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    __atomic_store_n(&g_driver, driver, __ATOMIC_RELEASE);
  }
  return driver ? dlsym(driver, name) : NULL;
}

/* ─── GPU UUIDs ───────────────────────────────────────────────────────────────
 * CUPTI names a GPU by device ordinal; the contract wants its UUID as
 * nvidia-smi prints it, "MIG-" and the instance's UUID for an instance of a
 * GPU split with MIG. Each is looked up once through the driver the process
 * already loaded, and never freed. An empty string marks a failed lookup. */
#define MAX_GPUS 64

static pthread_mutex_t g_gpu_mu = PTHREAD_MUTEX_INITIALIZER;
static char g_gpu_uuid[MAX_GPUS][sizeof("GPU-00000000-0000-0000-0000-000000000000")];
static const char *g_gpu_ptr[MAX_GPUS];

static void gpu_lookup_uuid(uint32_t dev)
{
  typedef CUresult (*device_get_fn)(CUdevice *, int);
  typedef CUresult (*device_uuid_fn)(CUuuid *, CUdevice);
  device_get_fn device_get     = (device_get_fn)driver_function("cuDeviceGet");
  /* v2 returns the UUID of a MIG instance, v1 always that of the whole GPU. */
  device_uuid_fn instance_uuid = (device_uuid_fn)driver_function("cuDeviceGetUuid_v2");
  device_uuid_fn gpu_uuid_v1   = (device_uuid_fn)driver_function("cuDeviceGetUuid");
  CUdevice device;
  if (!device_get || device_get(&device, (int)dev) != CUDA_SUCCESS) {
    return;
  }
  CUuuid uuid, whole;
  const int have_instance = instance_uuid && instance_uuid(&uuid, device) == CUDA_SUCCESS;
  const int have_whole    = gpu_uuid_v1 && gpu_uuid_v1(&whole, device) == CUDA_SUCCESS;
  if (!have_instance && !have_whole) {
    return;
  }
  if (!have_instance) {
    uuid = whole;
  }
  const int mig = have_instance && have_whole && memcmp(&uuid, &whole, sizeof(uuid)) != 0;
  /* 8-4-4-4-12 hex digits, as nvidia-smi prints it */
  char *out     = g_gpu_uuid[dev];
  out += sprintf(out, mig ? "MIG-" : "GPU-");
  for (size_t i = 0; i < sizeof(uuid.bytes); i++) {
    if (i == 4 || i == 6 || i == 8 || i == 10) {
      *out++ = '-';
    }
    out += sprintf(out, "%02x", (unsigned char)uuid.bytes[i]);
  }
}

/* Returns the UUID of GPU dev, or NULL if unknown. */
static const char *gpu_uuid(uint32_t dev)
{
  if (dev >= MAX_GPUS) {
    return NULL;
  }
  const char *uuid = __atomic_load_n(&g_gpu_ptr[dev], __ATOMIC_ACQUIRE);
  if (!uuid) {
    pthread_mutex_lock(&g_gpu_mu);
    if (!g_gpu_ptr[dev]) {
      gpu_lookup_uuid(dev);
      __atomic_store_n(&g_gpu_ptr[dev], g_gpu_uuid[dev], __ATOMIC_RELEASE);
    }
    uuid = g_gpu_ptr[dev];
    pthread_mutex_unlock(&g_gpu_mu);
  }
  return uuid[0] ? uuid : NULL;
}
