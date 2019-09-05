#ifndef __RKNN_THREAD_H__
#define __RKNN_THREAD_H__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/time.h>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>

#include "opencv2/core/core.hpp"

struct rknn_out_data {
	float *out[8];
};

struct rknn_queue_data {
	cv::Mat * img_origin;
	cv::Mat * img_resize;
	struct rknn_out_data out_data;
};

class rknn_opencv {
      private:
	int mat_count;
	bool is_running;
	float fps;
	int frame_count;
	unsigned long last_time;

	void *user_ctx_ptr;
	struct rknn_queue_data *mats_queue;
	 cv::Mat * mats_origin;
	 cv::Mat * mats_resize;
	 std::queue < struct rknn_queue_data *>idle_queue;
	 std::queue < struct rknn_queue_data *>input_queue;
	 std::queue < struct rknn_queue_data *>output_queue;
	 std::mutex mtx_idle;
	 std::mutex mtx_input;
	 std::mutex mtx_output;
	 std::condition_variable cond_input_not_empty;
	 std::condition_variable cond_output_not_empty;

	int (*get_img_func) (void *ctx, cv::Mat & orinig, cv::Mat & resize);
	int (*detect_img_func) (void *ctx, cv::Mat & resize,
				struct rknn_out_data * out_data);
	int (*show_img_func) (void *ctx, cv::Mat & orinig, float fps,
			      struct rknn_out_data * out_data);

	 std::thread thread_get_img;
	 std::thread thread_detect_img;
	 std::thread thread_show_img;

	static unsigned long get_time(void);

	static void *get_img_task(void *data);

	static void *detect_img_task(void *data);

      public:
	 rknn_opencv();
	~rknn_opencv();

	int start(int (*get_img)
		  (void *ctx, cv::Mat & orinig, cv::Mat & resize),
		  int (*detect_img) (void *ctx, cv::Mat & resize,
				     struct rknn_out_data * out_data),
		  int (*show_img) (void *ctx, cv::Mat & orinig, float fps,
				   struct rknn_out_data * out_data),
		  void *user_ctx);

	int update_show(void);

	int stop(void);
};

#endif
