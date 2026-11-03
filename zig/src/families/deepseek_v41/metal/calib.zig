//! The Metal backend's boot calibration (TF_DSV41_CALIB=measure, or zig with no stored table): window and pass costs.
const std = @import("std");
const host = @import("dsv41");
const target_ = @import("target.zig");
const pipelines = @import("pipelines.zig");

const calib_lanes = host.calib_lanes;
const calib_env = host.calib_env;
const iface = host.iface;
const ph = host.phases;
const Allocator = std.mem.Allocator;

/// What a measurement runs on: the loaded target, the drafter's pass and this rank's share of the work.
pub const Ctx = struct {
    a: Allocator,
    io: std.Io,
    t: *target_.Target,
    /// DSpark's pass (null: serial windows, no draft run)
    pass: ?iface.Pass,
    /// ranks a timed run waits for (1 on one device)
    world: u32 = 1,
    /// the prompt cache the run put aside, and its size to give back
    cache_cap: ?u64 = null,
};

/// `TF_DSV41_CALIB=measure` / `zig`: the calibration text through the slots, the plan's runs timed, the table stored.
pub fn boot(c: *Ctx, dir: []const u8, k: calib_env.Knobs, want: calib_env.Shape) !calib_lanes.Boot {
    const cfg = c.t.env.c;
    return calib_lanes.boot(c.a, c.io, .{
        .ctx = c,
        .target = c.t.target(),
        .pass = c.pass,
        .slots = @intCast(c.t.slots.len),
        .limit = c.t.opt.limit,
        .rows = pipelines.max_rows,
        .world = c.world,
        .vocab = cfg.vocab,
        .shape = shapeOf(cfg),
        .mark = mark,
        .gather = gather,
        .has_pending = hasPending,
        .drop_window = dropWindow,
        .detach = detach,
    }, dir, k, want);
}

/// DSpark's shape as serve.zig `model()` declares it: the release's, at most 64 candidates of the vocabulary.
pub fn shapeOf(c: *const host.config.Config) host.dspark.Shape {
    return .{ .block = c.dspark_block, .noise = c.dspark_noise_token, .window = c.window, .rank = c.dspark_markov_rank, .hidden = c.hidden, .candidates = @min(64, c.vocab) };
}

/// A run's end here: every Metal launch completes before it returns (forward.zig, dspark.zig), so the clock reads now.
fn mark(_: *anyopaque) anyerror!u64 {
    return ph.nowNs();
}

/// One rank's samples into the gather buffer (`world` 1: a copy).
fn gather(_: *anyopaque, ours: []const i64, all: []i64) anyerror!void {
    if (all.len != ours.len) return error.BadWindow;
    @memcpy(all, ours);
}

/// The slot's prefill left a verify tail pending (a window never kept).
fn hasPending(ctx: *anyopaque, slot: u32) bool {
    const c: *const Ctx = @ptrCast(@alignCast(ctx));
    return slot < c.t.slots.len and c.t.slots[slot].open;
}

/// The slot's pending window dropped, nothing kept.
fn dropWindow(ctx: *anyopaque, slot: u32) anyerror!void {
    const c: *const Ctx = @ptrCast(@alignCast(ctx));
    return c.t.dropWindow(slot);
}

/// The prompt cache out for the run and back after it: the calibration text is neither resumed from nor saved.
fn detach(ctx: *anyopaque) void {
    const c: *Ctx = @ptrCast(@alignCast(ctx));
    if (c.cache_cap) |cap| {
        c.t.cache.cap = cap;
        c.cache_cap = null;
        return;
    }
    c.cache_cap = c.t.cache.cap;
    c.t.cache.cap = 0;
}

test "the shape a measurement is stored under is the served model's" {
    var c = host.config.Config{ .vocab = 129280, .window = 128, .hidden = 64 };
    try std.testing.expectEqual(@as(u32, 64), shapeOf(&c).candidates);
    try std.testing.expectEqual(@as(u32, 128), shapeOf(&c).window);
    c.vocab = 16;
    try std.testing.expectEqual(@as(u32, 16), shapeOf(&c).candidates);
}
