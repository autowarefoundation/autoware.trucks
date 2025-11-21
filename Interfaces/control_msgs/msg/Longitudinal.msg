# Longitudinal control message
#
# Naming convention comply with SAE J1939
#
# Positive values represent forward motion (+X)
# Negative values represent backward motion (-X)

# Time this message was created
builtin_interfaces/Time stamp

# Time this configuration state is expected to be achieved in (optional)
builtin_interfaces/Time control_time

# Desired engine torque in % of engine reference torque
float32 engine_torque_cmd

# External brake request (XBR) in m/s² 
float32 external_brake_request

# XBR EBI mode
uint8 ebi_mode 

