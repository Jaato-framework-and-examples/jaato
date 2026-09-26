/**
 * The Files panel's image viewer shows the bytes it was handed as an image
 * from a blob URL, says what the image is, and lets it be zoomed.
 *
 * jsdom has no layout, so what is asserted here is what does not need one:
 * the blob carries the mime (an SVG without its type is not decoded as an
 * image), the URL is revoked, the info line reads the natural size once the
 * image loads, and a decode failure says so instead of showing a broken
 * image.  Zooming a real image is the e2e suite's job.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import ImageView, { describeImage } from "./ImageView";

// react-zoom-pan-pinch watches its wrapper's size; jsdom has no ResizeObserver.
globalThis.ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} } as unknown as typeof ResizeObserver;

let blobs: Blob[] = [];
beforeEach(() => {
  blobs = [];
  Object.assign(URL, {
    createObjectURL: vi.fn((b: Blob) => { blobs.push(b); return `blob:img-${blobs.length}`; }),
    revokeObjectURL: vi.fn(),
  });
});
afterEach(cleanup);

const bytes = new Uint8Array([60, 115, 118, 103, 47, 62]); // "<svg/>"

describe("ImageView", () => {
  it("shows the bytes from a typed blob URL and revokes it on unmount", () => {
    const { unmount } = render(<ImageView data={bytes} mime="image/svg+xml" label="logo.svg" heightClass="h-64" />);
    expect(screen.getByRole("img", { name: "logo.svg" })).toHaveAttribute("src", "blob:img-1");
    expect(blobs[0]!.type).toBe("image/svg+xml");
    unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:img-1");
  });

  it("reads the natural size once the image loads", () => {
    render(<ImageView data={bytes} mime="image/png" label="chart.png" heightClass="h-64" />);
    expect(screen.getByTestId("image-info")).toHaveTextContent("PNG · 6 B");
    const img = screen.getByRole("img", { name: "chart.png" });
    Object.defineProperty(img, "naturalWidth", { value: 1920 });
    Object.defineProperty(img, "naturalHeight", { value: 1080 });
    fireEvent.load(img);
    expect(screen.getByTestId("image-info")).toHaveTextContent("1920×1080 · PNG · 6 B");
  });

  it("says the image could not be decoded instead of showing a broken one", () => {
    render(<ImageView data={bytes} mime="image/png" label="broken.png" heightClass="h-64" />);
    fireEvent.error(screen.getByRole("img", { name: "broken.png" }));
    expect(screen.queryByRole("img")).toBeNull();
    expect(screen.getByText(/could not decode this image/)).toBeInTheDocument();
  });

  it("offers fit, 1:1 and zoom controls", () => {
    render(<ImageView data={bytes} mime="image/png" label="x.png" heightClass="h-64" />);
    for (const name of ["fit", "1:1", "Zoom in", "Zoom out"]) expect(screen.getByRole("button", { name })).toBeInTheDocument();
  });
});

describe("describeImage", () => {
  it("names the size, kind and weight it knows", () => {
    expect(describeImage({ w: 10, h: 20 }, "image/svg+xml", 2048)).toBe("10×20 · SVG · 2.0 KB");
    expect(describeImage(null, "image/jpeg", 3 * 1024 * 1024)).toBe("JPEG · 3.0 MB");
  });
});
