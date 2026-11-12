find_package(OpenCV REQUIRED)

add_executable(test_planning
        test_planning.cpp
)

#add_executable(test_longitudinal_planning
#        test_longitudinal_planning.cpp
#)
#
#add_executable(test_lateral_planning
#        test_lateral_planning.cpp
#)

target_link_libraries(test_planning
        PRIVATE
        planning
        ${OpenCV_LIBS}
)

#target_link_libraries(test_longitudinal_planning
#        PRIVATE
#            planning
#            ${OpenCV_LIBS}
#)
#
#target_link_libraries(test_lateral_planning
#        PRIVATE
#        planning
#        ${OpenCV_LIBS}
#)