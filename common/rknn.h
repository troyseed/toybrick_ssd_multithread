#ifndef __RKNN_H__
#define __RKNN_H__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include "rockchip/rknn_api.h"

class rknn {
private:
	int output_state;
	rknn_context rknn_ctx;
	std::vector <rknn_tensor_attr >attrs;
	std::vector <rknn_output >float_outs;
	rknn_input_output_num in_out_num;
	rknn_output_extend out_extend;

	int sync_output(void);
public:
	rknn();

	~rknn();

	int load_model(const char *model_path, uint32_t flag);

	int query(rknn_query_cmd cmd, void *info, int info_len);

	int set_input(int index, void *buf, int len);

	int run(rknn_run_extend *extend);

	int get_in_out_num(rknn_input_output_num *in_out);

	int get_outsize(int index, int *size);

	int get_outputs_data(int index, float *data, int size);

	int get_outputs_extend(rknn_output_extend *extend);
};

#endif
