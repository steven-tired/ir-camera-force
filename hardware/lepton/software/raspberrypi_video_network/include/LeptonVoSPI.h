#ifndef LEPTON_VOSPI_H
#define LEPTON_VOSPI_H

#include <cstddef>
#include <cstdint>

namespace lepton_vospi
{
constexpr std::size_t kPacketSize = 164;
constexpr std::size_t kImagePacketsPerSegment = 60;
constexpr std::size_t kTelemetryPacketsPerSegment = 61;
constexpr std::size_t kImageSegmentBytes = kPacketSize * kImagePacketsPerSegment;
constexpr std::size_t kTelemetrySegmentBytes = kPacketSize * kTelemetryPacketsPerSegment;
constexpr std::size_t kSegmentIdPacket = 20;
constexpr int kSegmentCount = 4;

inline bool supported_segment_size(std::size_t size)
{
    return size == kImageSegmentBytes || size == kTelemetrySegmentBytes;
}

inline int segment_number(const std::uint8_t *segment, std::size_t size)
{
    if (segment == nullptr || !supported_segment_size(size))
    {
        return -1;
    }
    return (segment[kSegmentIdPacket * kPacketSize] >> 4) & 0x0f;
}

inline bool validate_segment(const std::uint8_t *segment, std::size_t size)
{
    if (segment == nullptr || !supported_segment_size(size))
    {
        return false;
    }
    const std::size_t packet_count = size / kPacketSize;
    for (std::size_t packet = 0; packet < packet_count; ++packet)
    {
        const std::size_t offset = packet * kPacketSize;
        const std::uint16_t packet_number =
            (static_cast<std::uint16_t>(segment[offset] & 0x0f) << 8) | segment[offset + 1];
        if (packet_number != packet)
        {
            return false;
        }
    }
    const int number = segment_number(segment, size);
    return number >= 1 && number <= kSegmentCount;
}
} // namespace lepton_vospi

#endif
