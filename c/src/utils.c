#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>



typedef struct {
    int num_layers;
    float lr;
    int epochs;
    float momentum;
    int batch_size;
    int hidden_layer;
} Params;

struct ParamField {
    const char *name;
    size_t offset;
    char type;    // 'i' = int, 'f' = float
};

static const struct ParamField param_fields[] = {
    { "num_layers",     offsetof(Params, num_layers),     'i' },
    { "lr",             offsetof(Params, lr),             'f' },
    { "epochs",         offsetof(Params, epochs),         'i' },
    { "momentum",       offsetof(Params, momentum),       'f' },
    { "batch_size",     offsetof(Params, batch_size),     'i' },
    { "hidden_layer",   offsetof(Params, hidden_layer),   'i' },
};

static const int param_fields_count = sizeof(param_fields) / sizeof(param_fields[0]);






void load_params(const char *filename, Params *p) {
    printf("load_params: P0\n");
    FILE *f = fopen(filename, "r");
    printf("load_params: P00\n");
    if (!f) {
        perror(filename);
        return;
    }


    char key[128], value[128];

    while (fscanf(f, "%127s %127s", key, value) == 2) {
        printf("load_params: P1\n");
        for (int i = 0; i < param_fields_count; i++) {
            if (strcmp(key, param_fields[i].name) == 0) {

                char type = param_fields[i].type;
                void *field_ptr = (char *)p + param_fields[i].offset;

                if (type == 'i') {
                    *(int *)field_ptr = atoi(value);
                    printf("load_params: %s\n", param_fields[i].name);
                }
                else if (type == 'f') {
                    *(float *)field_ptr = atof(value);
                }
            }
        }
    }
    printf("load_params: P1\n");
    fclose(f);
}
