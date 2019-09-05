#include <chrono>
#include "rknn_thread.h"

static const std::chrono::milliseconds con_var_time_out(100);

unsigned long rknn_opencv::get_time(void)
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
	return (ts.tv_sec * 1000 + ts.tv_usec / 1000);
}

void *rknn_opencv::get_img_task(void *data)
{
	int ret;
	rknn_opencv *pd = (rknn_opencv *) data;
	printf("%s thread start...\n", __func__);

	while (pd->is_running) {
		cv::Mat tmp_origin;
		cv::Mat tmp_resize;

		ret = (*pd->get_img_func) (pd->user_ctx_ptr,
					   tmp_origin, tmp_resize);
		if (ret < 0) {
			printf("get_img_func error[%d], stop demo...\n", ret);
			pd->is_running = false;
		}

		std::unique_lock < std::mutex > lock(pd->mtx_idle);

		if (pd->idle_queue.empty()) {
			continue;
		}

		auto img = pd->idle_queue.front();
		pd->idle_queue.pop();
		lock.unlock();

		*(img->img_origin) = tmp_origin;
		*(img->img_resize) = tmp_resize;

		pd->mtx_input.lock();
		pd->input_queue.push(img);
		pd->cond_input_not_empty.notify_all();
		pd->mtx_input.unlock();
	}

	printf("%s thread terminate...\n", __func__);
	return NULL;
}

void *rknn_opencv::detect_img_task(void *data)
{
	int ret;
	rknn_opencv *pd = (rknn_opencv *) data;
	printf("%s thread start...\n", __func__);

	while (pd->is_running) {
		std::unique_lock < std::mutex > lock(pd->mtx_input);

		if (pd->input_queue.empty()) {
			pd->cond_input_not_empty.wait_for(lock,
							  con_var_time_out);
			continue;
		}

		auto img = pd->input_queue.front();
		pd->input_queue.pop();
		lock.unlock();

		ret = (*pd->detect_img_func) (pd->user_ctx_ptr,
					      *(img->img_resize),
					      &img->out_data);
		if (ret < 0) {
			printf("detect_img_func error[%d], stop demo...\n",
			       ret);
			pd->is_running = false;
		}

		pd->mtx_output.lock();
		pd->output_queue.push(img);
		pd->cond_output_not_empty.notify_all();
		pd->mtx_output.unlock();
	}

	printf("%s thread terminate...\n", __func__);
	return NULL;
}

rknn_opencv::rknn_opencv()
{
	mat_count = 8;
	is_running = false;
	fps = 0.0;
	user_ctx_ptr = NULL;

	mats_queue = new rknn_queue_data[mat_count];
	mats_origin = new cv::Mat[mat_count];
	mats_resize = new cv::Mat[mat_count];

	for (int i = 0; i < mat_count; i++) {
		mats_queue[i].img_origin = &mats_origin[i];
		mats_queue[i].img_resize = &mats_resize[i];
		idle_queue.push(&mats_queue[i]);
	}

	frame_count = 0;
}

rknn_opencv::~rknn_opencv()
{
	delete[]mats_origin;
	delete[]mats_resize;
	delete[]mats_queue;
}

int rknn_opencv::start(int (*get_img) (void *, cv::Mat &, cv::Mat &),
		       int (*detect_img) (void *, cv::Mat &,
					  struct rknn_out_data *),
		       int (*show_img) (void *, cv::Mat &, float,
					struct rknn_out_data *), void *user_ctx)
{
	if (!get_img || !detect_img || !show_img)
		return -1;

	get_img_func = get_img;
	detect_img_func = detect_img;
	show_img_func = show_img;
	user_ctx_ptr = user_ctx;

	is_running = true;
	last_time = get_time();
	thread_get_img = std::thread(get_img_task, this);
	thread_detect_img = std::thread(detect_img_task, this);

	return 0;
}

int rknn_opencv::update_show(void)
{
	int ret;
	std::unique_lock < std::mutex > lock(mtx_output);

	if (output_queue.empty()) {
		if (is_running) {
			cond_output_not_empty.wait_for(lock, con_var_time_out);
			return 0;
		} else {
			return -1;
		}
	}

	auto img = output_queue.front();
	output_queue.pop();
	lock.unlock();

	frame_count++;
	unsigned long cur_time = get_time();
	if (cur_time - last_time > 1000) {
		float sec_time = (cur_time - last_time) / 1000.0;
		fps = frame_count / sec_time;
		printf("%f, %5.2f\n", fps, fps);
		last_time = cur_time;
		frame_count = 0;
	}

	ret =
	    (*show_img_func) (user_ctx_ptr, *(img->img_origin), fps,
			      &img->out_data);

	mtx_idle.lock();
	idle_queue.push(img);
	mtx_idle.unlock();

	return ret;
}

int rknn_opencv::stop(void)
{
	is_running = false;
	usleep(10000);

	cond_input_not_empty.notify_all();
	cond_output_not_empty.notify_all();

	thread_get_img.join();
	thread_detect_img.join();

	return 0;
}
