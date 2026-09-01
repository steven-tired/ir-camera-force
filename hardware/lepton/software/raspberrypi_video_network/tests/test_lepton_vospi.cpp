#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "LeptonVoSPI.h"

int main()
{
    using namespace lepton_vospi;

    static_assert(kPacketSize == 164, "Lepton Raw14 packets are 164 bytes");
    static_assert(kImagePacketsPerSegment == 60, "image-only segment size changed");
    static_assert(kTelemetryPacketsPerSegment == 61, "telemetry segment size changed");

    std::vector<std::uint8_t> segment(kTelemetrySegmentBytes, 0);
    for (std::size_t packet = 0; packet < kTelemetryPacketsPerSegment; ++packet)
    {
        segment[packet * kPacketSize + 1] = static_cast<std::uint8_t>(packet);
    }
    segment[kSegmentIdPacket * kPacketSize] = 3 << 4;

    assert(validate_segment(segment.data(), segment.size()));
    assert(segment_number(segment.data(), segment.size()) == 3);

    segment[12 * kPacketSize + 1] = 13;
    assert(!validate_segment(segment.data(), segment.size()));
    assert(!validate_segment(segment.data(), kTelemetrySegmentBytes - 1));
    return 0;
}
