// Minimal Apple Silicon PMU reader via the private kperf/kperfdata frameworks
// (the interface Instruments' CPU Counters uses). Needs root.
// Pattern follows ibireme's kpc_demo.c (MIT).
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef uint64_t kpc_config_t;
typedef struct kpep_db kpep_db;
typedef struct kpep_config kpep_config;
typedef struct kpep_event kpep_event;
#define KPC_MAX_COUNTERS 32
#define KPC_CLASS_CONFIGURABLE_MASK (1u << 1)

static int (*kpc_set_counting)(uint32_t);
static int (*kpc_set_thread_counting)(uint32_t);
static int (*kpc_set_config)(uint32_t, kpc_config_t *);
static int (*kpc_get_thread_counters)(uint32_t, uint32_t, uint64_t *);
static int (*kpc_force_all_ctrs_set)(int);
static int (*kpep_config_create)(kpep_db *, kpep_config **);
static int (*kpep_config_add_event)(kpep_config *, kpep_event **, uint32_t, uint32_t *);
static int (*kpep_config_force_counters)(kpep_config *);
static int (*kpep_config_kpc)(kpep_config *, kpc_config_t *, size_t);
static int (*kpep_config_kpc_count)(kpep_config *, size_t *);
static int (*kpep_config_kpc_classes)(kpep_config *, uint32_t *);
static int (*kpep_config_kpc_map)(kpep_config *, size_t *, size_t);
static int (*kpep_db_create)(const char *, kpep_db **);
static int (*kpep_db_event)(kpep_db *, const char *, kpep_event **);

#define GSIZE 6
static const char *GROUPS[][GSIZE] = {
    {"FIXED_CYCLES", "FIXED_INSTRUCTIONS", "RETIRE_UOP", "INST_SIMD_ALU", "INST_SIMD_LD", "INST_SIMD_ST"},
    {"FIXED_CYCLES", "FIXED_INSTRUCTIONS", "INST_INT_ALU", "INST_INT_LD", "INST_INT_ST", "INST_BRANCH"},
    {"FIXED_CYCLES", "FIXED_INSTRUCTIONS", "L1D_CACHE_MISS_LD", "L1D_CACHE_MISS_ST", "BRANCH_MISPRED_NONSPEC", "L1D_TLB_MISS"},
    {"FIXED_CYCLES", "FIXED_INSTRUCTIONS", "ARM_STALL_BACKEND", "ARM_STALL_FRONTEND", "MAP_STALL_DISPATCH", "SCHEDULE_EMPTY"},
    {"FIXED_CYCLES", "FIXED_INSTRUCTIONS", "LD_UNIT_WAITING_YOUNG_L1D_CACHE_MISS", "ST_MEM_ORDER_VIOL_LD_NONSPEC", "LD_SRC_STORE_NONSPEC", "PL2_CACHE_MISS_LD"},
    {"FIXED_CYCLES", "FIXED_INSTRUCTIONS", "MAP_SIMD_UOP", "MAP_LDST_UOP", "MAP_INT_UOP", "MAP_UOP"},
    {"FIXED_CYCLES", "FIXED_INSTRUCTIONS", "INST_INT_ST", "INST_LDST", "INST_SIMD_ALU_VEC", "ARM_STALL_SLOT_BACKEND"},
};
#define NGROUPS (int)(sizeof(GROUPS) / sizeof(GROUPS[0]))

static size_t g_map[KPC_MAX_COUNTERS];
static int g_n;
static int g_ok[GSIZE];  // 1 if event i was configured
static int g_slot[GSIZE]; // index into kpc map for event i

#define LOAD(lib, sym) do { *(void **)&sym = dlsym(lib, #sym); if (!sym) { fprintf(stderr, "kpc: missing %s\n", #sym); return -1; } } while (0)

int kc_num_groups(void) { return NGROUPS; }
int kc_group_size(int g) { return GSIZE; }
const char *kc_event_name(int g, int i) { return GROUPS[g][i]; }

static int load(void) {
    static void *kp, *kd;
    if (kp) return 0;
    kp = dlopen("/System/Library/PrivateFrameworks/kperf.framework/kperf", RTLD_LAZY);
    kd = dlopen("/System/Library/PrivateFrameworks/kperfdata.framework/kperfdata", RTLD_LAZY);
    if (!kp || !kd) { fprintf(stderr, "kpc: dlopen failed\n"); return -1; }
    LOAD(kp, kpc_set_counting); LOAD(kp, kpc_set_thread_counting); LOAD(kp, kpc_set_config);
    LOAD(kp, kpc_get_thread_counters); LOAD(kp, kpc_force_all_ctrs_set);
    LOAD(kd, kpep_config_create); LOAD(kd, kpep_config_add_event); LOAD(kd, kpep_config_force_counters);
    LOAD(kd, kpep_config_kpc); LOAD(kd, kpep_config_kpc_count); LOAD(kd, kpep_config_kpc_classes);
    LOAD(kd, kpep_config_kpc_map); LOAD(kd, kpep_db_create); LOAD(kd, kpep_db_event);
    return 0;
}

static uint32_t g_classes; static size_t g_nreg; static kpc_config_t g_regs[KPC_MAX_COUNTERS];

// Build the event config for a group (no root needed). Events that cannot be
// scheduled alongside the earlier ones are skipped (g_ok[i] = 0).
static int build(int group) {
    if (load()) return -1;
    if (group < 0 || group >= NGROUPS) return -2;
    kpep_db *db = 0; kpep_config *cfg = 0;
    if (kpep_db_create(NULL, &db) || !db) { fprintf(stderr, "kpc: no PMU database for this CPU\n"); return -4; }
    if (kpep_config_create(db, &cfg) || kpep_config_force_counters(cfg)) return -5;
    int added = 0;
    for (int i = 0; i < GSIZE; i++) {
        g_ok[i] = 0;
        kpep_event *ev = 0;
        if (kpep_db_event(db, GROUPS[group][i], &ev) || !ev) continue;
        uint32_t err = 0;
        if (kpep_config_add_event(cfg, &ev, 0, &err)) continue;
        g_ok[i] = 1; g_slot[i] = added++;
    }
    memset(g_regs, 0, sizeof(g_regs));
    if (kpep_config_kpc_classes(cfg, &g_classes) || kpep_config_kpc_count(cfg, &g_nreg)) return -8;
    if (kpep_config_kpc_map(cfg, g_map, sizeof(g_map)) || kpep_config_kpc(cfg, g_regs, sizeof(g_regs))) return -9;
    return added;
}

// Dry run (no root): bitmask of events in the group that can be scheduled together.
int kc_check(int group) {
    int n = build(group); if (n < 0) return n;
    int mask = 0; for (int i = 0; i < GSIZE; i++) mask |= g_ok[i] << i;
    return mask;
}

int kc_event_ok(int i) { return g_ok[i]; }

// Configure and start counting. Returns number of events configured, or negative on error.
int kc_init(int group) {
    int n = build(group); if (n < 0) return n;
    if (kpc_force_all_ctrs_set(1)) { fprintf(stderr, "kpc: permission denied (run with sudo)\n"); return -3; }
    if ((g_classes & KPC_CLASS_CONFIGURABLE_MASK) && g_nreg && kpc_set_config(g_classes, g_regs)) return -10;
    if (kpc_set_counting(g_classes) || kpc_set_thread_counting(g_classes)) return -11;
    g_n = GSIZE;
    return n;
}

// Writes the configured events' current thread counts into out[0..n).
void kc_read(uint64_t *out) {
    uint64_t c[KPC_MAX_COUNTERS] = {0};
    kpc_get_thread_counters(0, KPC_MAX_COUNTERS, c);
    for (int i = 0; i < g_n; i++) out[i] = g_ok[i] ? c[g_map[g_slot[i]]] : 0;
}
