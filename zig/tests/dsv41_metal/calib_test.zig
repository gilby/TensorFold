//! The Metal backend's draft pricing: TF_DSV41_CALIB's choice, the measurement at open, its stored table.
const std = @import("std");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fs = @import("fixture_serve.zig");
const ff = @import("fixture_fwd.zig");

const api = host.api;
const calib_env = host.calib_env;
const Served = host.serve_lanes.Served;
const testing = std.testing;
const gpa = testing.allocator;
const io = testing.io;
const seed = 71;
const platform = @import("sys");

/// A reply's tokens and finish, written on LaneHost's thread and read on the test's.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    out: std.ArrayList(u32) = .empty,
    done: bool = false,
    reason: ?api.Reason = null,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .tokens => |ts| b.out.appendSlice(gpa, ts) catch {},
            .finished => |fin| {
                b.done = true;
                b.reason = fin.reason;
            },
            else => {},
        }
    }

    fn wait(b: *Box) !void {
        for (0..600_000) |_| {
            b.mutex.lockUncancelable(io);
            const done = b.done;
            b.mutex.unlock(io);
            if (done) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};

/// 8 tokens of a reply from `e` (gpa-owned).
fn reply(e: api.Engine, prompt: []const u32, drafts: bool) ![]u32 {
    var b: Box = .{};
    defer b.out.deinit(gpa);
    const r: api.Request = .{ .prompt = prompt, .max_tokens = 8, .drafts = drafts };
    try e.submit(1, &r, .{ .ctx = &b, .event = Box.event });
    try b.wait();
    try testing.expectEqual(api.Reason.length, b.reason.?);
    return b.out.toOwnedSlice(gpa);
}

/// The cell the tiny model's table is stored under: its 4 slots, one device, DSpark, its context limit.
fn shape() calib_env.Shape {
    return .{ .slots = 4, .world = 1, .dspark = true, .context = fs.limit };
}

test "dsv41 metal calib: the served model prices drafts from TF_DSV41_CALIB (unset: zig)" {
    const m = try fs.open(gpa, io, seed, .{ .calib = null });
    defer fs.close(m);
    const saved = std.c.getenv("TF_DSV41_CALIB");
    defer if (saved) |v| {
        _ = platform.env.setenv("TF_DSV41_CALIB", v, 1);
    } else {
        _ = platform.env.unsetenv("TF_DSV41_CALIB");
    };
    _ = platform.env.unsetenv("TF_DSV41_CALIB");
    try testing.expectEqualStrings("zig", m.model().calib.?.mode.?);
    _ = platform.env.setenv("TF_DSV41_CALIB", "Measure", 1);
    try testing.expectEqualStrings("Measure", m.model().calib.?.mode.?);
    _ = platform.env.setenv("TF_DSV41_CALIB", "  ", 1); // blank: the server's own default
    try testing.expectEqualStrings("zig", m.model().calib.?.mode.?);
    _ = platform.env.setenv("TF_DSV41_CALIB", "default", 1);
    try testing.expectEqualStrings("default", m.model().calib.?.mode.?);
}

test "dsv41 metal calib: zig measures once, stores the table of its shape, and the next open reads it back" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    const knobs: calib_env.Knobs = .{ .mode = "zig", .dir = dir };
    // nothing stored yet: the open has to measure
    var none = try calib_env.load(gpa, io, knobs, shape(), 64);
    defer none.deinit(gpa);
    try testing.expectEqual(calib_env.Source.measure, none.source);
    const m = try fs.open(gpa, io, seed, .{ .calib = knobs });
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    const verify = try gpa.dupe(f64, s.costs.verify);
    defer gpa.free(verify);
    // a measured table, not the shipped one's 42 ms a row (this model's row 1 is ~1 ms)
    try testing.expect(verify.len >= 16);
    try testing.expect(verify[0] > 0.0 and verify[0] < 20.0);
    for (1..verify.len) |i| try testing.expect(verify[i] >= verify[i - 1]);
    try testing.expect(s.costs.draft > 0.0 and s.costs.draft < 5.0);
    // calibration prices drafts only: the replies are the serial ones, drafted or not
    const prompt = try ff.ids(gpa, &fs.config(), 40, 11);
    defer gpa.free(prompt);
    const ref = try fs.open(gpa, io, seed, .{ .slots = 1, .drafts = false });
    defer fs.close(ref);
    const want = try ref.t.serial(0, prompt, 8, null);
    defer gpa.free(want);
    const drafted = try reply(s.engine(), prompt, true);
    defer gpa.free(drafted);
    const plain = try reply(s.engine(), prompt, false);
    defer gpa.free(plain);
    try testing.expectEqualSlices(u32, want, drafted);
    try testing.expectEqualSlices(u32, want, plain);
    // stored in the cache dir as this engine's entry for this shape
    var stored = try calib_env.load(gpa, io, knobs, shape(), 64);
    defer stored.deinit(gpa);
    try testing.expectEqual(calib_env.Source.cached, stored.source);
    var nb: [64]u8 = undefined;
    try testing.expectEqualStrings(try calib_env.zigName(&nb, shape()), std.fs.path.basename(stored.path.?));
    for (verify, stored.costs.verify[0..verify.len]) |a, b| try testing.expect(@abs(a - b) <= 0.002);
    // the next open of the same shape reads that entry instead of measuring again
    const m2 = try fs.open(gpa, io, seed, .{ .calib = knobs });
    const s2 = try Served.open(gpa, io, m2.model(), ".", null);
    defer Served.close(s2);
    try testing.expectEqualSlices(f64, stored.costs.verify, s2.costs.verify);
    try testing.expectEqual(stored.costs.draft, s2.costs.draft);
    std.debug.print("PASS dsv41 metal calib: {d} rows measured and stored, row 1 {d:.3} / 4 {d:.3} / 8 {d:.3} / 16 {d:.3} / 64 {d:.3} ms, pass {d:.3} ms, slot {d:.3} ms, drafted == plain == serial; the next open read them back\n", .{ verify.len, verify[0], verify[3], verify[7], verify[15], verify[63], s.costs.draft, s.costs.slot });
}

test "dsv41 metal calib: measure takes a fresh table even where one is stored" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    var nb: [64]u8 = undefined;
    const name = try calib_env.zigName(&nb, shape());
    try tmp.dir.writeFile(io, .{ .sub_path = name, .data =
        \\{"verify": [999.0, 999.0], "draft": 999.0, "slot": 0.0, "engine": "zig",
        \\ "meta": {"shape": {"slots": 4, "world": 1, "dspark": true, "context": 1280}, "time": "2026-01-01T00:00:00"}}
    });
    const m = try fs.open(gpa, io, seed, .{ .calib = .{ .mode = "measure", .dir = dir } });
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    try testing.expect(s.costs.verify[0] < 20.0);
    try testing.expect(s.costs.draft < 5.0);
    // the fresh table replaced the stored one: a zig open now reads a measurement's, not 999
    var stored = try calib_env.load(gpa, io, .{ .mode = "zig", .dir = dir }, shape(), 64);
    defer stored.deinit(gpa);
    try testing.expectEqual(calib_env.Source.cached, stored.source);
    try testing.expect(stored.costs.verify[0] < 20.0);
}

test "dsv41 metal calib: default keeps the shipped table; an unknown mode is refused by name" {
    const m = try fs.open(gpa, io, seed, .{ .calib = .{ .mode = "default" } });
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    try testing.expectEqual(@as(f64, 42.0), s.costs.verify[0]);
    try testing.expectEqual(@as(usize, 64), s.costs.verify.len);
    try testing.expectEqual(@as(f64, 5.6), s.costs.draft);
    const bad = try fs.open(gpa, io, seed, .{ .calib = .{ .mode = "fastest", .dir = "." } });
    const bad_model = bad.model();
    defer fs.close(bad);
    try testing.expectError(error.CalibMode, Served.open(gpa, io, bad_model, ".", null));
}
