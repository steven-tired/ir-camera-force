#include "ThermalFrameAssembler.h"

namespace thermal_stream {

ThermalFrameAssembler::Result ThermalFrameAssembler::add_datagram(
    const std::uint8_t *data,
    std::size_t size,
    std::vector<std::uint16_t> &completed_frame)
{
    if (data == nullptr || size != kDatagramBytes || !validate_packets(data))
    {
        reset();
        return Result::Rejected;
    }

    const std::size_t segment_header = kSegmentHeaderPacket * kPacketBytes;
    const std::uint8_t segment = static_cast<std::uint8_t>(data[segment_header] >> 4);
    if (segment != expected_segment_)
    {
        reset();
        if (segment != 1)
        {
            return Result::Rejected;
        }
    }

    const std::size_t packets_to_copy = segment == 4 ? 57 : kPacketsPerSegment;
    partial_frame_.reserve(kImagePixels);
    for (std::size_t packet = 0; packet < packets_to_copy; ++packet)
    {
        const std::size_t payload = packet * kPacketBytes + kHeaderBytes;
        for (std::size_t pixel = 0; pixel < kPixelsPerPacket; ++pixel)
        {
            const std::size_t offset = payload + pixel * 2;
            partial_frame_.push_back(static_cast<std::uint16_t>(
                (static_cast<std::uint16_t>(data[offset]) << 8) | data[offset + 1]));
        }
    }

    if (segment != 4)
    {
        ++expected_segment_;
        return Result::Incomplete;
    }

    completed_frame.swap(partial_frame_);
    reset();
    return completed_frame.size() == kImagePixels ? Result::Complete : Result::Rejected;
}

bool ThermalFrameAssembler::validate_packets(const std::uint8_t *data) const
{
    for (std::size_t packet = 0; packet < kPacketsPerSegment; ++packet)
    {
        const std::size_t offset = packet * kPacketBytes;
        const std::uint16_t packet_number = static_cast<std::uint16_t>(
            (static_cast<std::uint16_t>(data[offset] & 0x0f) << 8) | data[offset + 1]);
        if (packet_number != packet)
        {
            return false;
        }

        if (packet == kSegmentHeaderPacket)
        {
            const std::uint8_t segment = static_cast<std::uint8_t>(data[offset] >> 4);
            if (segment < 1 || segment > 4)
            {
                return false;
            }
        }
        else if ((data[offset] >> 4) == 0x0f)
        {
            return false;
        }
    }
    return true;
}

void ThermalFrameAssembler::reset()
{
    expected_segment_ = 1;
    partial_frame_.clear();
}

std::array<PixelRange, 4> renderable_pixel_ranges(
    const std::vector<std::uint16_t> &frame)
{
    const std::size_t full_segment_pixels =
        ThermalFrameAssembler::kPacketsPerSegment * ThermalFrameAssembler::kPixelsPerPacket;
    const std::array<std::size_t, 4> segment_pixels = {{
        full_segment_pixels,
        full_segment_pixels,
        full_segment_pixels,
        (ThermalFrameAssembler::kImagePackets -
         3 * ThermalFrameAssembler::kPacketsPerSegment) *
            ThermalFrameAssembler::kPixelsPerPacket
    }};

    std::array<PixelRange, 4> ranges;
    std::size_t segment_begin = 0;
    for (std::size_t segment = 0; segment < ranges.size(); ++segment)
    {
        const std::size_t segment_end = segment_begin + segment_pixels[segment];
        ranges[segment] = {segment_begin, segment_end};
        for (std::size_t pixel = segment_begin; pixel < segment_end; ++pixel)
        {
            if (frame[pixel] == 0)
            {
                ranges[segment].end = pixel;
                break;
            }
        }
        segment_begin = segment_end;
    }
    return ranges;
}

} // namespace thermal_stream
