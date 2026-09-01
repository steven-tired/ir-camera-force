#include "ThermalFrameAssembler.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

using thermal_stream::ThermalFrameAssembler;
using Result = ThermalFrameAssembler::Result;

std::vector<std::uint8_t> make_segment(std::uint8_t segment_number,
                                       std::uint16_t frame_marker)
{
    std::vector<std::uint8_t> datagram(ThermalFrameAssembler::kDatagramBytes);
    for (std::size_t packet = 0;
         packet < ThermalFrameAssembler::kPacketsPerSegment;
         ++packet)
    {
        const std::size_t offset = packet * ThermalFrameAssembler::kPacketBytes;
        datagram[offset] = packet == ThermalFrameAssembler::kSegmentHeaderPacket
                               ? static_cast<std::uint8_t>(segment_number << 4)
                               : 0;
        datagram[offset + 1] = static_cast<std::uint8_t>(packet);

        const std::size_t global_packet =
            (static_cast<std::size_t>(segment_number) - 1) *
                ThermalFrameAssembler::kPacketsPerSegment +
            packet;
        for (std::size_t pixel = 0;
             pixel < ThermalFrameAssembler::kPixelsPerPacket;
             ++pixel)
        {
            const std::uint16_t value = static_cast<std::uint16_t>(
                frame_marker + global_packet * ThermalFrameAssembler::kPixelsPerPacket + pixel);
            datagram[offset + ThermalFrameAssembler::kHeaderBytes + pixel * 2] =
                static_cast<std::uint8_t>(value >> 8);
            datagram[offset + ThermalFrameAssembler::kHeaderBytes + pixel * 2 + 1] =
                static_cast<std::uint8_t>(value & 0xff);
        }
    }
    return datagram;
}

void require(bool condition, const char *message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

bool submit_frame(ThermalFrameAssembler &assembler,
                  std::uint16_t marker,
                  std::vector<std::uint16_t> &pixels)
{
    Result result = Result::Rejected;
    for (std::uint8_t segment = 1; segment <= 4; ++segment)
    {
        const auto datagram = make_segment(segment, marker);
        result = assembler.add_datagram(datagram.data(), datagram.size(), pixels);
        require(result == (segment == 4 ? Result::Complete : Result::Incomplete),
                "frame completed at the wrong segment");
    }
    return result == Result::Complete;
}

void test_reconstructs_rows_and_excludes_footer()
{
    ThermalFrameAssembler assembler;
    std::vector<std::uint16_t> pixels;
    const std::uint16_t marker = 1000;

    require(submit_frame(assembler, marker, pixels), "valid frame was rejected");
    require(pixels.size() == ThermalFrameAssembler::kImagePixels,
            "output was not exactly 160x120 pixels");

    require(pixels[0] == marker, "row 0 left half was reconstructed incorrectly");
    require(pixels[79] == marker + 79, "row 0 left edge was reconstructed incorrectly");
    require(pixels[80] == marker + 80, "row 0 right half was reconstructed incorrectly");
    require(pixels[159] == marker + 159, "row 0 right edge was reconstructed incorrectly");
    require(pixels[160] == marker + 160, "row 1 was reconstructed incorrectly");
    require(pixels.back() == marker + ThermalFrameAssembler::kImagePixels - 1,
            "footer packets were rendered or the last image packet was lost");
}

void test_rejects_invalid_inputs_and_recovers()
{
    ThermalFrameAssembler assembler;
    std::vector<std::uint16_t> pixels;

    auto wrong_size = make_segment(1, 2000);
    require(assembler.add_datagram(wrong_size.data(), wrong_size.size() - 1, pixels) ==
                Result::Rejected,
            "wrong datagram size was accepted");

    auto bad_packet = make_segment(1, 2000);
    bad_packet[17 * ThermalFrameAssembler::kPacketBytes + 1] = 18;
    require(assembler.add_datagram(bad_packet.data(), bad_packet.size(), pixels) ==
                Result::Rejected,
            "wrong packet sequence was accepted");

    auto segment_two = make_segment(2, 2000);
    require(assembler.add_datagram(segment_two.data(), segment_two.size(), pixels) ==
                Result::Rejected,
            "out-of-order segment was accepted");

    auto segment_one = make_segment(1, 2000);
    require(assembler.add_datagram(segment_one.data(), segment_one.size(), pixels) ==
                Result::Incomplete,
            "partial frame completed early");
    Result recovered = Result::Rejected;
    for (std::uint8_t segment = 1; segment <= 4; ++segment)
    {
        const auto datagram = make_segment(segment, 3000);
        recovered = assembler.add_datagram(datagram.data(), datagram.size(), pixels);
        require(recovered == (segment == 4 ? Result::Complete : Result::Incomplete),
                "assembler did not recover on the next valid 1-2-3-4 frame");
    }
    require(pixels.front() == 3000, "recovered frame contains stale pixels");
}

void test_accepts_live_non_discard_high_nibbles()
{
    ThermalFrameAssembler assembler;
    std::vector<std::uint16_t> pixels;
    Result result = Result::Rejected;

    const std::uint8_t live_high_nibbles[] = {0x10, 0x30, 0x40};
    for (std::uint8_t segment = 1; segment <= 4; ++segment)
    {
        auto datagram = make_segment(segment, 4000);
        for (std::size_t packet = 0;
             packet < ThermalFrameAssembler::kPacketsPerSegment;
             ++packet)
        {
            if (packet == ThermalFrameAssembler::kSegmentHeaderPacket)
            {
                continue;
            }
            datagram[packet * ThermalFrameAssembler::kPacketBytes] =
                live_high_nibbles[(packet + segment) % 3];
        }

        result = assembler.add_datagram(datagram.data(), datagram.size(), pixels);
        require(result == (segment == 4 ? Result::Complete : Result::Incomplete),
                "valid live high nibbles prevented frame completion");
    }
    require(pixels.front() == 4000 &&
                pixels.size() == ThermalFrameAssembler::kImagePixels,
            "live-layout frame pixels were not assembled");
}

void test_rejects_discard_and_illegal_segment_headers()
{
    ThermalFrameAssembler assembler;
    std::vector<std::uint16_t> pixels;

    auto discard_packet = make_segment(1, 4000);
    discard_packet[0] = 0xf0;
    require(assembler.add_datagram(discard_packet.data(), discard_packet.size(), pixels) ==
                Result::Rejected,
            "VoSPI discard header was accepted as packet 0");

    auto illegal_segment_header = make_segment(1, 4000);
    illegal_segment_header[ThermalFrameAssembler::kSegmentHeaderPacket *
                           ThermalFrameAssembler::kPacketBytes] = 0x90;
    require(assembler.add_datagram(illegal_segment_header.data(),
                                   illegal_segment_header.size(), pixels) == Result::Rejected,
            "illegal packet-20 segment encoding was accepted");
}

void test_zero_stops_only_the_current_segment_render_prefix()
{
    std::vector<std::uint16_t> pixels(ThermalFrameAssembler::kImagePixels, 1);
    const std::size_t full_segment_pixels =
        ThermalFrameAssembler::kPacketsPerSegment * ThermalFrameAssembler::kPixelsPerPacket;
    pixels[5] = 0;
    pixels[full_segment_pixels + 7] = 0;
    pixels[3 * full_segment_pixels] = 0;

    const auto ranges = thermal_stream::renderable_pixel_ranges(pixels);
    require(ranges[0].begin == 0 && ranges[0].end == 5,
            "segment 1 did not stop at its first zero");
    require(ranges[1].begin == full_segment_pixels &&
                ranges[1].end == full_segment_pixels + 7,
            "segment 2 did not resume and stop at its own first zero");
    require(ranges[2].begin == 2 * full_segment_pixels &&
                ranges[2].end == 3 * full_segment_pixels,
            "zero-free segment 3 was not rendered completely");
    require(ranges[3].begin == 3 * full_segment_pixels &&
                ranges[3].end == 3 * full_segment_pixels,
            "segment 4 did not stop when its first pixel was zero");
}

} // namespace

int main()
{
    test_reconstructs_rows_and_excludes_footer();
    test_rejects_invalid_inputs_and_recovers();
    test_accepts_live_non_discard_high_nibbles();
    test_rejects_discard_and_illegal_segment_headers();
    test_zero_stops_only_the_current_segment_render_prefix();
    std::cout << "thermal_frame_assembler_test: PASS\n";
    return EXIT_SUCCESS;
}
