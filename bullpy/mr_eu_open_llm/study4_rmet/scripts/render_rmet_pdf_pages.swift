#!/usr/bin/env swift
// Render each page of an RMET PDF to a PNG using macOS PDFKit.
// study4-only helper; does not depend on mr_eu_open_llm / study3 code.
import Foundation
import PDFKit
import AppKit

guard CommandLine.arguments.count >= 3 else {
    fputs("Usage: render_rmet_pdf_pages.swift <input.pdf> <output_dir> [dpi]\n", stderr)
    exit(2)
}

let inputPath = CommandLine.arguments[1]
let outputDir = CommandLine.arguments[2]
let dpi = CommandLine.arguments.count >= 4 ? CGFloat(Double(CommandLine.arguments[3]) ?? 150.0) : 150.0

guard let doc = PDFDocument(url: URL(fileURLWithPath: inputPath)) else {
    fputs("Failed to open PDF: \(inputPath)\n", stderr)
    exit(1)
}

try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

let scale = dpi / 72.0
for i in 0..<doc.pageCount {
    guard let page = doc.page(at: i) else { continue }
    let bounds = page.bounds(for: .mediaBox)
    let width = Int((bounds.width * scale).rounded())
    let height = Int((bounds.height * scale).rounded())
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        fputs("Failed to create context for page \(i)\n", stderr)
        continue
    }
    ctx.setFillColor(NSColor.white.cgColor)
    ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
    ctx.saveGState()
    ctx.translateBy(x: 0, y: CGFloat(height))
    ctx.scaleBy(x: scale, y: -scale)
    page.draw(with: .mediaBox, to: ctx)
    ctx.restoreGState()
    guard let cgImage = ctx.makeImage() else { continue }
    let rep = NSBitmapImageRep(cgImage: cgImage)
    guard let data = rep.representation(using: .png, properties: [:]) else { continue }
    let outPath = (outputDir as NSString).appendingPathComponent(String(format: "page_%03d.png", i + 1))
    try data.write(to: URL(fileURLWithPath: outPath))
    fputs("wrote \(outPath)\n", stderr)
}
fputs("done pages=\(doc.pageCount)\n", stderr)
