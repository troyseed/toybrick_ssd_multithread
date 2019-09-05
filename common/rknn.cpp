#include "rknn.h"
#include <cstring>

#define ASSERT_RKNN_CTX(rknn_ctx) 	do {\
						if (rknn_ctx < 0)\
							return RKNN_ERR_MODEL_INVALID;\
					} while (0)
#define CHECK_NULL(x) 	do {\
				if (!(x))\
					return RKNN_ERR_PARAM_INVALID;\
			} while (0)
enum {
	OUT_INVALID = 0,
	OUT_SYNC,
	OUT_OKAY,
};

rknn::rknn()
{
	output_state = OUT_INVALID;
	rknn_ctx = 0;
}

rknn::~rknn()
{
	if (rknn_ctx != 0)
		rknn_destroy(rknn_ctx);

		for (auto i:float_outs) {
			free(i.buf);
	}
}

int rknn::sync_output(void)
{
	if (output_state == OUT_SYNC) {
		int ret;
		rknn_output outputs[in_out_num.n_output];

		memset(&outputs, 0x00, sizeof(outputs));
		for (int i = 0; i < in_out_num.n_output; i++) {
			outputs[i].want_float  = true;
			outputs[i].is_prealloc = true;
			outputs[i].index       = i;
			outputs[i].buf         = float_outs[i].buf;
			outputs[i].size        = float_outs[i].size;
		}

		ret = rknn_outputs_get(rknn_ctx, in_out_num.n_output, outputs,
				       &out_extend);
		if (ret < 0)
			return ret;

		ret = rknn_outputs_release(rknn_ctx, in_out_num.n_output, outputs);
		if (ret < 0)
			return ret;

		output_state = OUT_OKAY;
	}

	return 0;
}

int rknn::load_model(const char *model_path, uint32_t flag)
{
	int ret = 0;
	FILE *fp = NULL;
	int model_len = 0;
	unsigned char *model = NULL;

	CHECK_NULL(model_path);

	/* Load model to bufffer */
	printf("loading model...\n");
	fp = fopen(model_path, "rb");
	if (fp == NULL) {
		printf("fopen %s fail!\n", model_path);
		ret = RKNN_ERR_MODEL_INVALID;
		goto exit;
	}

	fseek(fp, 0, SEEK_END);
	model_len = ftell(fp);
	model = (unsigned char *)malloc(model_len);
	if (!model) {
		printf("Malloc %d buffer fail!\n", model_len);
		ret = RKNN_ERR_MALLOC_FAIL;
		goto exit;
	}

	fseek(fp, 0, SEEK_SET);
	if (model_len != fread(model, 1, model_len, fp)) {
		printf("fread %s fail!\n", model_path);
		goto exit;
	}

	/* load model to npu */
	ret = rknn_init(&rknn_ctx, model, model_len, flag);
	if (ret < 0) {
		printf("rknn_init fail! ret=%d\n", ret);
		goto exit;
	}

	ret = rknn_query(rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &in_out_num,
			 sizeof(in_out_num));
	if (ret)
		goto exit;

	printf("input num = %u, output = num = %u\n",
	       in_out_num.n_input, in_out_num.n_output);

	for (int i = 0; i < in_out_num.n_output; i++) {
		rknn_tensor_attr tmp_attr;
		tmp_attr.index = i;
		ret = rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR,
				 &tmp_attr, sizeof(tmp_attr));
		if (ret < 0) {
			goto exit;
		}
		printf("tensor index %d is %s, type = %02x, qnt_type = %02x\n",
		       i, tmp_attr.name, tmp_attr.type, tmp_attr.qnt_type);
		attrs.push_back(tmp_attr);

		rknn_output out;
		memset(&out, 0x00, sizeof(out));
		out.index = i;
		out.size = tmp_attr.n_elems * sizeof(float);

		float *data = (float *)malloc(out.size);
		memset(data, 0x00, out.size);
		out.buf = data;

		float_outs.push_back(out);
	}

      exit:
	if (fp)
		fclose(fp);
	if (model)
		free(model);

	return ret;
}

int rknn::query(rknn_query_cmd cmd, void *info, int info_len)
{
	CHECK_NULL(info);

	ASSERT_RKNN_CTX(rknn_ctx);

	return rknn_query(rknn_ctx, cmd, info, info_len);
}

int rknn::set_input(int index, void *buf, int len)
{
	rknn_input input;

	CHECK_NULL(buf);

	ASSERT_RKNN_CTX(rknn_ctx);

	memset(&input, 0x00, sizeof(input));
	input.index = 0;
	input.buf   = buf;
	input.size  = len;
	input.pass_through = false;
	input.type  = RKNN_TENSOR_UINT8;
	input.fmt   = RKNN_TENSOR_NHWC;

	return rknn_inputs_set(rknn_ctx, 1, &input);
}

int rknn::run(rknn_run_extend *extend)
{
	int ret;

	ASSERT_RKNN_CTX(rknn_ctx);

	ret = rknn_run(rknn_ctx, extend);
	if (ret == 0)
		output_state = OUT_SYNC;

	return ret;
}

int rknn::get_in_out_num(rknn_input_output_num *in_out)
{
	CHECK_NULL(in_out);

	memset(in_out, 0x00, sizeof(*in_out));

	ASSERT_RKNN_CTX(rknn_ctx);

	in_out->n_input = in_out_num.n_input;
	in_out->n_output = in_out_num.n_output;

	return 0;
}

int rknn::get_outsize(int index, int *size)
{

	CHECK_NULL(size);

	*size = 0;

	ASSERT_RKNN_CTX(rknn_ctx);

	if (index > float_outs.size())
		return RKNN_ERR_OUTPUT_INVALID;

	*size = float_outs[index].size;

	return 0;
}

int rknn::get_outputs_data(int index, float *data, int size)
{
	CHECK_NULL(data);

	ASSERT_RKNN_CTX(rknn_ctx);

	if (size < float_outs[index].size)
		return RKNN_ERR_PARAM_INVALID;

	if (index >= in_out_num.n_output || output_state == OUT_INVALID)
		return RKNN_ERR_OUTPUT_INVALID;

	sync_output();

	memcpy(data, float_outs[index].buf, float_outs[index].size);

	return 0;
}

int rknn::get_outputs_extend(rknn_output_extend *extend)
{
	ASSERT_RKNN_CTX(rknn_ctx);

	if (output_state == OUT_INVALID)
		return RKNN_ERR_OUTPUT_INVALID;

	sync_output();

	memcpy(extend, &out_extend, sizeof(*extend));

	return 0;
}
