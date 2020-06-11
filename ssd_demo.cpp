#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include "rknn_test.h"

using namespace std;
using namespace cv;

#define NUM_RESULTS	1917
#define NUM_CLASSES	91

static const int VIDEO_NODE = 0;
static const char *WIN_NAME = "ssd";
static const char *MODEL_PATH = "../models/ssd/mobilenet_ssd.rknn";

static const int INPUT_WIDTH = 300;
static const int INPUT_HEIGHT = 300;
static const int INPUT_CHANNEL = 3;

static const char *LABEL_PATH = "../models/ssd/coco_labels_list.txt";
static const char *BOX_PRIO_PATH = "../models/ssd/box_priors.txt";

#define Y_SCALE  10.0f
#define X_SCALE  10.0f
#define H_SCALE  5.0f
#define W_SCALE  5.0f

struct ssd_data {
	float boxPriors[4][NUM_RESULTS];
	string labels[91];
};

Scalar colorArray[10] = {
	Scalar(139, 0, 0, 255),
	Scalar(139, 0, 139, 255),
	Scalar(0, 0, 139, 255),
	Scalar(0, 100, 0, 255),
	Scalar(139, 139, 0, 255),
	Scalar(209, 206, 0, 255),
	Scalar(0, 127, 255, 255),
	Scalar(139, 61, 72, 255),
	Scalar(0, 255, 0, 255),
	Scalar(255, 0, 0, 255),
};

float MIN_SCORE = 0.6f;
float NMS_THRESHOLD = 0.45f;

const char *get_valid_file(const char *file)
{
	int ret;
	const char *file_name;

	ret = access(file, F_OK);
	if (ret) {
		file_name = strrchr(file, '/') + 1;
		ret = access(file_name, F_OK);
		if (ret)
			return NULL;
		else
			return file_name;
	} else {
		return file;
	}
}

int loadLabelName(string locationFilename, string * labels)
{
	ifstream fin(locationFilename);
	string line;
	int lineNum = 0;

	while (getline(fin, line)) {
		labels[lineNum] = line;
		lineNum++;
	}
	return 0;
}

int loadCoderOptions(string locationFilename, float (*boxPriors)[NUM_RESULTS])
{
	const char *d = ", ";
	ifstream fin(locationFilename);
	string line;
	int lineNum = 0;

	while (getline(fin, line)) {
		char *line_str = const_cast < char *>(line.c_str());
		char *p;
		p = strtok(line_str, d);
		int priorIndex = 0;

		while (p) {
			float number = static_cast < float >(atof(p));
			boxPriors[lineNum][priorIndex++] = number;
			p = strtok(nullptr, d);
		}

		if (priorIndex != NUM_RESULTS) {
			return -1;
		}

		lineNum++;
	}

	return 0;

}

float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0,
		       float xmin1, float ymin1, float xmax1, float ymax1)
{
	float w = max(0.f, min(xmax0, xmax1) - max(xmin0, xmin1));
	float h = max(0.f, min(ymax0, ymax1) - max(ymin0, ymin1));
	float i = w * h;
	float u = (xmax0 - xmin0) * (ymax0 - ymin0) +
	    (xmax1 - xmin1) * (ymax1 - ymin1) - i;
	return u <= 0.f ? 0.f : (i / u);
}

float expit(float x)
{
	return (float)(1.0 / (1.0 + exp(-x)));
}

void decodeCenterSizeBox(float *predictions, float (*boxPriors)[NUM_RESULTS], int index)
{
	int base = index * 4;

	float ycenter =
	    predictions[base + 0] / Y_SCALE * boxPriors[2][index] +
	    boxPriors[0][index];
	float xcenter =
	    predictions[base + 1] / X_SCALE * boxPriors[3][index] +
	    boxPriors[1][index];
	float h =
	    (float)exp(predictions[base + 2] / H_SCALE) *
	    boxPriors[2][index];
	float w =
	    (float)exp(predictions[base + 3] / W_SCALE) *
	    boxPriors[3][index];

	float ymin = ycenter - h / 2.0f;
	float xmin = xcenter - w / 2.0f;
	float ymax = ycenter + h / 2.0f;
	float xmax = xcenter + w / 2.0f;

	predictions[base + 0] = ymin;
	predictions[base + 1] = xmin;
	predictions[base + 2] = ymax;
	predictions[base + 3] = xmax;

}

int scaleToInputSize(float *outputClasses, int (*output)[NUM_RESULTS],
		     int numClasses)
{
	int validCount = 0;

	// Scale them back to the input size.
	for (int i = 0; i < NUM_RESULTS; ++i) {
		float topClassScore = static_cast < float >(-1000.0);
		int topClassScoreIndex = -1;

		// Skip the first catch-all class.
		for (int j = 1; j < numClasses; ++j) {
			float score = outputClasses[i * numClasses + j];
			if (score > topClassScore) {
				topClassScoreIndex = j;
				topClassScore = score;
			}
		}

		topClassScore = expit(topClassScore);
		if (topClassScore >= MIN_SCORE) {
			output[0][validCount] = i;
			output[1][validCount] = topClassScoreIndex;
			++validCount;
		}
	}

	return validCount;
}

int nms(int validCount, float *outputLocations, int (*output)[NUM_RESULTS])
{
	for (int i = 0; i < validCount; ++i) {
		if (output[0][i] == -1) {
			continue;
		}

		int n = output[0][i];
		for (int j = i + 1; j < validCount; ++j) {
			int m = output[0][j];
			if (m == -1) {
				continue;
			}
			float xmin0 = outputLocations[n * 4 + 1];
			float ymin0 = outputLocations[n * 4 + 0];
			float xmax0 = outputLocations[n * 4 + 3];
			float ymax0 = outputLocations[n * 4 + 2];

			float xmin1 = outputLocations[m * 4 + 1];
			float ymin1 = outputLocations[m * 4 + 0];
			float xmax1 = outputLocations[m * 4 + 3];
			float ymax1 = outputLocations[m * 4 + 2];

			float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0,
						     xmin1, ymin1, xmax1,
						     ymax1);

			if (iou >= NMS_THRESHOLD && output[1][i] == output[1][j]) {
				output[0][j] = -1;
			}
		}
	}

	return 0;
}

int post_process(void *data, cv::Mat & img, float fps,
		 struct rknn_out_data *out_data)
{
	char tmp_buf[32];
	int output[2][NUM_RESULTS];
	struct ssd_data *ssd = (struct ssd_data *)data;

	sprintf(tmp_buf, "FPS:%5.2f", fps);

	cv::putText(img, tmp_buf, Point(50, 50), 1, 3, Scalar(0, 0, 255, 255), 3);

	/* transform */
	float *predictions = (float *)out_data->out[1];
	float *outputClasses = (float *)out_data->out[0];
	if (predictions && outputClasses) {
		int validCount =
		    scaleToInputSize(outputClasses, output, NUM_CLASSES);

		for (int i; i < validCount; i++)
			decodeCenterSizeBox(predictions, ssd->boxPriors, output[0][i]);

		if (validCount < 100) {
			/* detect nest box */
			nms(validCount, predictions, output);
			/* box valid detect target */
			for (int i = 0; i < validCount; ++i) {
				if (output[0][i] == -1) {
					continue;
				}

				int n = output[0][i];
				int topClassScoreIndex = output[1][i];

				int x1 =
				    static_cast <
				    int >(predictions[n * 4 + 1] * img.cols);
				int y1 =
				    static_cast <
				    int >(predictions[n * 4 + 0] * img.rows);
				int x2 =
				    static_cast <
				    int >(predictions[n * 4 + 3] * img.cols);
				int y2 =
				    static_cast <
				    int >(predictions[n * 4 + 2] * img.rows);

				string label = ssd->labels[topClassScoreIndex];

				cv::rectangle(img, Point(x1, y1), Point(x2, y2),
					      colorArray[topClassScoreIndex %
							 10], 2);
				cv::putText(img, label, Point(x1, y1 + 20), 1, 2,
					    colorArray[topClassScoreIndex %
						       10], 2);
			}
		} else {
			printf("validCount too much!\n");
		}
	}

	return 0;
}

int main(void)
{
	int ret;
	const char *lable_path;
	const char *box_path;
	const char *model_path;
	struct ssd_data data;
	/* load label and boxPriors */
	lable_path = get_valid_file(LABEL_PATH);
	if (!lable_path) {
		printf("need lable file.\n");
		return -1;
	}

	box_path = get_valid_file(BOX_PRIO_PATH);
	if (!box_path) {
		printf("need box priors file.\n");
		return -1;
	}

	model_path = get_valid_file(MODEL_PATH);
	if (!model_path) {
		printf("need model file.\n");
		return -1;
	}

	loadLabelName(lable_path, data.labels);
	loadCoderOptions(box_path, data.boxPriors);

	class rknn_test test(WIN_NAME);

	/* 加载模型 */
	ret = test.load_model(model_path);
	if (ret < 0) {
		printf("load_model error!!!\n");
		return ret;
	}

	/* 设置输入图像的属性 */
	ret = test.set_input_info(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNEL);
	if (ret < 0) {
		printf("set_input_info error!!!\n");
		return ret;
	}

	/* 开始运行，支持摄像头和视频文件方式 */
	/* 使用摄像头时，VIDEO_NODE可以设置为0，表示从video 0节点获取图像数据 */
	/* 使用视频时，VIDEO_NODE可以设置为文件路径，如"xxx.mp4"，表示从视频文件获取图像数据 */
	/* post_process为自定义的后处理函数，RKNN相关操作已封装，只需完成后处理即可 */
	ret = test.run(VIDEO_NODE, post_process, &data);

	return ret;
}
