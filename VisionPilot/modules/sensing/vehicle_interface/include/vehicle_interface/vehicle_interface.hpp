#ifndef VISIONPILOT_VEHICLE_INTERFACE_HPP
#define VISIONPILOT_VEHICLE_INTERFACE_HPP


class VehicleInterface
{
public:
    VehicleInterface() = default;
    virtual ~VehicleInterface() = default;

    // Read vehicle speed via CAN frame
    virtual double read() = 0;

    // Send steering and acceleration via CAN frame
    virtual void write(double steering, double acceleration) = 0;
};


#endif //VISIONPILOT_VEHICLE_INTERFACE_HPP
