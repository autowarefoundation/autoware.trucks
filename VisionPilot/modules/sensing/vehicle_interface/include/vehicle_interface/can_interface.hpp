#ifndef VISIONPILOT_CAN_INTERFACE_HPP
#define VISIONPILOT_CAN_INTERFACE_HPP

#include <vehicle_interface/vehicle_interface.hpp>

class CanInterface : public VehicleInterface
{
public:
    CanInterface() = default;
    ~CanInterface() override = default;

    // Read vehicle speed via CAN frame
    double read() override;

    // Send steering and acceleration via CAN frame
    void write(double steering, double acceleration) override;
};

#endif //VISIONPILOT_CAN_INTERFACE_HPP
