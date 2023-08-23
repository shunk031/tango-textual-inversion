local model_name = 'runwayml/stable-diffusion-v1-5';

local placeholder_token = '<cat-toy>';
local initializer_token = 'toy';
local learnable_property = 'object';

local seed = 19950815;
local resolution = 512;
local batch_size = 4;
local grad_accum = 4;
local train_steps = 3000;
local devices = 1;
local output_dir = 'outputs/textual_inversion_model';

{
    include_package: [
        'textual_inversion',
    ],
    steps: {
        setup_tokenizer: {
            type: 'textual_inversion::setup_tokenizer',
            tokenizer: {
                pretrained_model_name_or_path: model_name,
                subfolder: 'tokenizer',
            },
            placeholder_token: placeholder_token,
        },
        raw_data: {
            type: 'datasets::load',
            path: 'diffusers/cat_toy_example',
        },
        transform_data: {
            type: 'textual_inversion::transform_data',
            dataset: { type: 'ref', ref: 'raw_data' },
            example_transformer: {
                type: 'textual_inversion',
                image_transformer: {
                    type: 'textual_inversion',
                    image_size: resolution,
                },
                prompt_transformer: {
                    type: 'textual_inversion',
                    placeholder_string: placeholder_token,
                    tokenizer: { type: 'ref', ref: 'setup_tokenizer' },
                    learnable_property: learnable_property,
                },
            },
        },
        trained_model: {
            type: 'torch::train',
            seed: seed,
            training_engine: {
                type: 'textual_inversion',
                optimizer: {
                    type: 'torch::AdamW',
                    lr: 5e-4,
                    betas: [0.9, 0.999],
                    weight_decay: 0.0,
                    eps: 1e-08,
                },
                lr_scheduler: {
                    type: 'diffusers::constant',
                },
            },
            model: {
                type: 'textual_inversion::stable_diffusion',
                model_name: model_name,
                tokenizer: { type: 'ref', ref: 'setup_tokenizer' },
                placeholder_token: placeholder_token,
                initializer_token: initializer_token,
            },
            dataset_dict: {
                type: 'ref',
                ref: 'transform_data',
            },
            train_dataloader: {
                shuffle: true,
                batch_size: batch_size,
                collate_fn: {
                    type: 'textual_inversion::custom_collator',
                },
            },
            train_steps: train_steps,
            grad_accum: grad_accum,
            device_count: devices,
            checkpoint_every: 1000,
        },
        create_pipeline: {
            type: 'textual_inversion::create_pipeline',
            model_name: model_name,
            model: { type: 'ref', ref: 'trained_model' },
            output_dir: output_dir,
            placeholder_token: placeholder_token,
        },
        generate_images: {
            type: 'textual_inversion::generate_images',
            pipe: { type: 'ref', ref: 'create_pipeline' },
            prompt: 'A <cat-toy> backpack',
            seed: seed,
            generated_image_path: 'cat-backpack.png',
            grid_rows: 2,
            grid_cols: 4,
        },
    },
}
