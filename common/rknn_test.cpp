#include "rknn_test.h"

rknn_test::rknn_test(const char *test_name)
{
	win_name = test_name;

	is_model_load = false;
	output_size = NULL;

	input_index = 0;
	num_output = 0;
	img_width = 0;
	img_height = 0;
	img_channels = 0;

	detect_time = 0;
	detect_count = 0;
	show_time = 0;
	show_count = 0;

	user_data = NULL;
}

rknn_test::~rknn_test(void)
{
	if (num_output) {
		delete[]output_size;
	}

	if (cap.isOpened())
		cap.release();
}

unsigned long rknn_test::get_time(void)
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
	return (ts.tv_sec * 1000 + ts.tv_usec / 1000);
}

int rknn_test::get_img(void *ctx, cv::Mat & origin, cv::Mat & resize)
{
	class rknn_test *test_ctx = (class rknn_test *) ctx;

	test_ctx->cap >> origin;
	if (origin.empty() || !origin.data)
		return -1;

	if (origin.cols != test_ctx->img_width
	    || origin.rows != test_ctx->img_height)
		cv::resize(origin, resize,
			   cv::Size(test_ctx->img_width, test_ctx->img_height),
			   (0, 0), (0, 0), cv::INTER_LINEAR);
	else
		resize = origin.clone();
	cv::cvtColor(resize, resize, cv::COLOR_BGR2RGB);

	return 0;
}

int rknn_test::show_img(void *ctx, cv::Mat & img, float fps,
			struct rknn_out_data *out_data)
{
	int ret;

	class rknn_test *test_ctx = (class rknn_test *) ctx;
	char tmp_buf[32];
	unsigned long start_time, end_time;

	start_time = get_time();

	ret = test_ctx->post_process(test_ctx->user_data, img, fps, out_data);

	imshow(test_ctx->win_name, img);

	end_time = get_time();

	test_ctx->show_count++;

	test_ctx->show_time += (end_time - start_time);

	if (test_ctx->show_count > 100) {
		printf("show_time = %ldms\n", test_ctx->show_time / 100);
		test_ctx->show_count = 0;
		test_ctx->show_time = 0;
	}

	for (int i = 0; i < ARRAY_SIZE(out_data->out); i++) {
		if (out_data->out[i])
			free(out_data->out[i]);
	}

	memset(out_data, 0x00, sizeof(*out_data));

	return 0;
}

int rknn_test::detect_img(void *ctx, cv::Mat & img,
			  struct rknn_out_data *out_data)
{
	int ret;
	class rknn_test *test_ctx = (class rknn_test *) ctx;
	unsigned long start_time, end_time;

	memset(out_data, 0x00, sizeof(*out_data));

	start_time = get_time();

	ret = test_ctx->rknn_api.set_input(test_ctx->input_index, img.data,
					   test_ctx->img_width *
					   test_ctx->img_height *
					   test_ctx->img_channels);
	if (ret < 0) {
		printf("rknn_input_set fail! ret=%d\n", ret);
		goto Error;
	}

	ret = test_ctx->rknn_api.run(nullptr);
	if (ret < 0) {
		printf("rknn_run fail! ret=%d\n", ret);
		goto Error;
	}

	// Process output
	for (int i = 0; i < test_ctx->num_output; i++) {
		out_data->out[i] = (float *)malloc(test_ctx->output_size[i]);

		test_ctx->rknn_api.get_outputs_data(i, out_data->out[i],
						    test_ctx->output_size[i]);
	}

	end_time = get_time();
	test_ctx->detect_count++;
	test_ctx->detect_time += (end_time - start_time);

	if (test_ctx->detect_count > 100) {
		printf("detect_time = %ldms\n", test_ctx->detect_time / 100);
		test_ctx->detect_count = 0;
		test_ctx->detect_time = 0;
	}

	return 0;

      Error:
	return -1;
}

int rknn_test::load_model(const char *path)
{
	int ret;
	rknn_input_output_num in_out_num;

	ret = rknn_api.load_model(path, RKNN_FLAG_PRIOR_MEDIUM);
	if (ret < 0)
		return ret;

	rknn_api.get_in_out_num(&in_out_num);
	num_output = in_out_num.n_output;
	output_size = new int[num_output];

	for (int i = 0; i < num_output; i++) {
		int size;
		rknn_api.get_outsize(i, &size);
		output_size[i] = size;
	}

	is_model_load = true;

	return 0;
}

int rknn_test::set_input_info(int width, int height, int channels)
{
	if (!is_model_load) {
		printf("should load model first!\n");
		return -1;
	}

	img_width = width;
	img_height = height;
	img_channels = channels;

	return 0;
}

int rknn_test::
common_run(int (*func)
	   (void *data, cv::Mat & img, float fps,
	    struct rknn_out_data * out_data), void *data)
{
	if (!is_model_load) {
		printf("should load model first!\n");
		return -1;
	}

	if (!img_width || !img_height || !img_channels) {
		printf("should set input info first!\n");
		return -1;
	}

	post_process = func;
	user_data = data;

	cv::Mat pre_img;
	cap >> pre_img;

	cv::imshow(win_name, pre_img);
	cv::waitKey(10);

	int ret, key;
	rknn_opencv test_quee;

	test_quee.start(get_img, detect_img, show_img, this);

	printf(">>>>>>>>>>>>>>>>>>press esc to exit<<<<<<<<<<<<<<<<<<<\n");

	while (1) {
		ret = test_quee.update_show();
		if (ret < 0) {
			printf("something is error!!!\n");
			break;
		}

		key = cv::waitKey(1);
		/* esc key */
		if ((key & 0xff) == 27) {
			printf("esc to skip...\n");
			break;
		}
	}

	test_quee.stop();

	cv::destroyAllWindows();

	return 0;
}

int rknn_test::run(int video_node,
		   int (*func) (void *data, cv::Mat & img, float fps,
				struct rknn_out_data * out_data), void *data)
{
	int ret;

	cap.open(video_node);
	if (!cap.isOpened()) {
		printf("open video[%d] error!!!\n", video_node);
		return -1;
	}

	ret = common_run(func, data);

	cap.release();

	return ret;
}

int rknn_test::run(const char *video_name,
		   int (*func) (void *data, cv::Mat & img, float fps,
				struct rknn_out_data * out_data), void *data)
{
	int ret;

	cap.open(video_name);
	if (!cap.isOpened()) {
		printf("open video [%s] error!!!\n", video_name);
		return -1;
	}

	ret = common_run(func, data);

	cap.release();

	return ret;
}
