#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <CLI/CLI.hpp>
#include <zenoh.h>

#include "masks_visualization_engine.hpp"

using namespace cv; 
using namespace std; 

#define DEFAULT_KEYEXPR "video/raw"
#define SEGMENTATION_KEYEXPR "video/segmented"

#define RECV_BUFFER_SIZE 100

z_owned_slice_t decode_frame_from_sample(const z_owned_sample_t& sample, int& row, int& col, int& type) {
    const z_loaned_sample_t* loaned_sample = z_loan(sample);
    z_owned_slice_t zslice;
    if (Z_OK != z_bytes_to_slice(z_sample_payload(loaned_sample), &zslice)) {
        throw std::runtime_error("Wrong payload");
    }

    // Extract the frame information for the attachment
    const z_loaned_bytes_t* attachment = z_sample_attachment(loaned_sample);
    if (attachment != NULL) {
        z_owned_slice_t output_bytes;
        int attachment_arg[3];
        z_bytes_to_slice(attachment, &output_bytes);
        memcpy(attachment_arg, z_slice_data(z_loan(output_bytes)), z_slice_len(z_loan(output_bytes)));
        row = attachment_arg[0];
        col = attachment_arg[1];
        type = attachment_arg[2];
        z_drop(z_move(output_bytes));
    } else {
        z_drop(z_move(zslice));
        throw std::runtime_error("No attachment");
    }

    // Return the slice, ownership is transferred to the caller.
    return zslice;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    CLI::App app{"Zenoh video subscriber example"};
    // Add options
    std::string keyexpr = DEFAULT_KEYEXPR;
    app.add_option("-k,--key", keyexpr, "The key expression to subscribe to")->default_val(DEFAULT_KEYEXPR);
    std::string seg_keyexpr = SEGMENTATION_KEYEXPR;
    app.add_option("-s,--segkey", seg_keyexpr, "The segmentation key expression to subscribe to")->default_val(SEGMENTATION_KEYEXPR);
    CLI11_PARSE(app, argc, argv);

    try {
        // Create Zenoh session
        z_owned_config_t config;
        z_owned_session_t s;
        z_config_default(&config);
        if (z_open(&s, z_move(config), NULL) < 0) {
            throw std::runtime_error("Error opening Zenoh session");
        }

        // Declare a Zenoh subscriber
        z_owned_subscriber_t sub;
        z_view_keyexpr_t ke;
        z_view_keyexpr_from_str(&ke, keyexpr.c_str());
        z_owned_ring_handler_sample_t handler;
        z_owned_closure_sample_t closure;
        z_ring_channel_sample_new(&closure, &handler, RECV_BUFFER_SIZE);
        if (z_declare_subscriber(z_loan(s), &sub, z_loan(ke), z_move(closure), NULL) < 0) {
            throw std::runtime_error("Error declaring Zenoh subscriber for key expression: " + std::string(keyexpr));
        }
        // Declare a Zenoh segmentation subscriber
        z_owned_subscriber_t seg_sub;
        z_view_keyexpr_t seg_ke;
        z_view_keyexpr_from_str(&seg_ke, seg_keyexpr.c_str());
        z_owned_ring_handler_sample_t seg_handler;
        z_owned_closure_sample_t seg_closure;
        z_ring_channel_sample_new(&seg_closure, &seg_handler, RECV_BUFFER_SIZE);
        if (z_declare_subscriber(z_loan(s), &seg_sub, z_loan(seg_ke), z_move(seg_closure), NULL) < 0) {
            throw std::runtime_error("Error declaring Zenoh subscriber for key expression: " + std::string(seg_keyexpr));
        }
        
        std::cout << "Subscribing to '" << keyexpr << "'..." << std::endl;
        std::cout << "Processing video... Press ESC to stop." << std::endl;
        z_owned_sample_t sample;
        while (Z_OK == z_recv(z_loan(handler), &sample)) {
            int row, col, type;
            z_owned_slice_t zslice = decode_frame_from_sample(sample, row, col, type);
            const uint8_t* ptr = z_slice_data(z_loan(zslice));
            // Release sample
            z_drop(z_move(sample));
            // Create the frame
            cv::Mat frame(row, col, type, const_cast<uint8_t*>(ptr));

            // Also receive segmentation frame
            if (Z_OK != z_try_recv(z_loan(seg_handler), &sample)) {
                std::cerr << "Warning: No segmentation frame received for the current video frame." << std::endl;
                z_drop(z_move(zslice)); // Release the slice after using its data pointer
                continue;
            }
            z_owned_slice_t seg_zslice = decode_frame_from_sample(sample, row, col, type);
            const uint8_t* seg_ptr = z_slice_data(z_loan(seg_zslice));
            // Release sample
            z_drop(z_move(sample));
            // Create the frame
            cv::Mat seg_frame(row, col, type, const_cast<uint8_t*>(seg_ptr));

            std::unique_ptr<autoware_pov::common::MasksVisualizationEngine> viz_engine_ = 
                std::make_unique<autoware_pov::common::MasksVisualizationEngine>("scene");
            cv::Mat final_frame = viz_engine_->visualize(seg_frame, frame);

            // Release the slice after using its data pointer
            z_drop(z_move(zslice));
            z_drop(z_move(seg_zslice));

            if (cv::waitKey(1) == 27) { // Stop if 'ESC' is pressed
                std::cout << "Processing stopped by user." << std::endl;
                break;
            }

            // Print frame rate
            static int frame_count = 0;
            static auto start_time = std::chrono::steady_clock::now();
            frame_count++;
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            if (elapsed_time > 0) {
                double fps = static_cast<double>(frame_count) / elapsed_time;
                std::cout << "Current FPS: " << fps << std::endl;
                frame_count = 0;
                start_time = current_time;
            }
        }

        // Clean up
        z_drop(z_move(handler));
        z_drop(z_move(sub));
        z_drop(z_move(seg_handler));
        z_drop(z_move(seg_sub));
        z_drop(z_move(s));
        cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
} 