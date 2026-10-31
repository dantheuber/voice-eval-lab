import { afterEach, describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { _resetHotkeys, useHotkeys } from "./useHotkeys";

const press = (key: string, init: KeyboardEventInit = {}) => window.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true, ...init }));

describe("useHotkeys", () => {
  afterEach(() => _resetHotkeys());

  it("fires a chord exactly once with two listeners in the same scope", () => {
    const a = vi.fn(), b = vi.fn();
    const h1 = renderHook(() => useHotkeys({ "g o": a }));
    const h2 = renderHook(() => useHotkeys({ "g o": b, j: vi.fn() }));
    press("g"); press("o");
    expect(a.mock.calls.length + b.mock.calls.length).toBe(1);
    h1.unmount(); h2.unmount();
  });

  it("does not let a chord prefix swallow a direct key owned by another listener", () => {
    const j = vi.fn();
    const h1 = renderHook(() => useHotkeys({ "g o": vi.fn() }));
    const h2 = renderHook(() => useHotkeys({ j }));
    press("j");
    expect(j).toHaveBeenCalledTimes(1);
    h1.unmount(); h2.unmount();
  });

  it("ignores letters while typing in an input but allows Ctrl combos", () => {
    const y = vi.fn(), save = vi.fn();
    const h = renderHook(() => useHotkeys({ y, "Ctrl+s": save }));
    const input = document.createElement("input"); document.body.appendChild(input); input.focus();
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "y", bubbles: true }));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "s", ctrlKey: true, bubbles: true }));
    expect(y).not.toHaveBeenCalled();
    expect(save).toHaveBeenCalledTimes(1);
    input.remove(); h.unmount();
  });
});
