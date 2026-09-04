#ifndef THERMAL_FRAME_ASSEMBLER_H
#define THERMAL_FRAME_ASSEMBLER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace thermal_stream {

struct PixelRange
{
    std::size_t begin;
    std::size_t end;
};

class ThermalFrameAssembler
{
public:
    enum class Result
    {
        Rejected,
        Incomplete,
        Complete
    };

    static constexpr std::size_t kPacketBytes = 164;
    static constexpr std::size_t kHeaderBytes = 4;
    static constexpr std::size_t kPixelsPerPacket = 80;
    static constexpr std::size_t kPacketsPerSegment = 61;
    static constexpr std::size_t kDatagramBytes = kPacketBytes * kPacketsPerSegment;
    static constexpr std::size_t kSegmentHeaderPacket = 20;
    static constexpr std::size_t kImagePackets = 240;
    static constexpr std::size_t kImagePixels = 160 * 120;

    Result add_datagram(const std::uint8_t *data,
                        std::size_t size,
                        std::vector<std::uint16_t> &completed_frame);

private:
    void reset();
    bool validate_packets(const std::uint8_t *data) const;

    std::uint8_t expected_segment_ = 1;
    std::vector<std::uint16_t> partial_frame_;
};

std::array<PixelRange, 4> renderable_pixel_ranges(
    const std::vector<std::uint16_t> &frame);

} // namespace thermal_stream

#endif
