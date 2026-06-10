/* SPDX-License-Identifier: Apache-2.0
 *
 * Minimal companion definitions for the vendored profiler_v2.h (from
 * NVIDIA/nccl plugins/profiler/example, Apache-2.0): the event-type bits and
 * state typedef live in NCCL's profiler.h, which drags in every ABI version —
 * only what v2 needs is defined here.
 */
#ifndef PROFILER_V2_COMPAT_H_
#define PROFILER_V2_COMPAT_H_

#include "common.h"
#include "err.h"

enum {
  ncclProfileGroup     = (1 << 0),
  ncclProfileColl      = (1 << 1),
  ncclProfileP2p       = (1 << 2),
  ncclProfileProxyOp   = (1 << 3),
  ncclProfileProxyStep = (1 << 4),
  ncclProfileProxyCtrl = (1 << 5),
};

typedef int ncclProfilerEventState_v2_t;

#include "profiler_v2.h"

#endif
