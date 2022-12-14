local model_name = "CompVis/stable-diffusion-v1-4";

local placeholder_token = "<cat-toy>";
local initializer_token = "toy";

local seed = 19950815;
local resolution = 512;
local batch_size = 4;
local grad_accum = 4;
local train_steps = 3000;
local devices = 1;
local output_dir = "outputs/textual_inversion_model";

{
    steps: {
        setup_tokenizer: {
            type: "setup_tokenizer",
            tokenizer: {
                type: "clip",
                pretrained_model_name_or_path: model_name,
                subfolder: "tokenizer",
            },
            placeholder_token: placeholder_token,
        },
        raw_data: {
            type: "datasets::load",
            path: "imagefolder",
            data_dir: "datasets/textual-inversion",
            drop_labels: true,
        },
        transform_data: {
            type: "transform_data",
            dataset: { type: "ref", ref: "raw_data" },
            tokenizer: { type: "ref", ref: "setup_tokenizer" },
            placeholder_token: placeholder_token,
            image_size: resolution,
        },
        trained_model: {
            type: "torch::train",
            seed: seed,
            training_engine: {
                type: "textual_inversion",
                optimizer: {
                    type: "torch::AdamW",
                    lr: 5e-4,
                    betas: [0.9, 0.999],
                    weight_decay: 0.0,
                    eps: 1e-08,
                },
                lr_scheduler: {
                    type: "diffusers::constant",
                },
            },
            model: {
                type: "stable_diffusion",
                model_name: model_name,
                tokenizer: { type: "ref", ref: "setup_tokenizer" },
                placeholder_token: placeholder_token,
                initializer_token: initializer_token,
            },
            dataset_dict: {
                type: "ref",
                ref: "transform_data",
            },
            train_dataloader: {
                shuffle: true,
                batch_size: batch_size,
                collate_fn: {
                    type: "custom_collator",
                },
            },
            train_steps: train_steps,
            grad_accum: grad_accum,
            device_count: devices,
            checkpoint_every: 1000,
        },
        save_pipeline: {
            type: "save_pipeline",
            model: { type: "ref", ref: "trained_model" },
            safety_checker_name: "CompVis/stable-diffusion-safety-checker",
            feature_extractor_name: "openai/clip-vit-base-patch32",
            output_dir: output_dir,
            placeholder_token: placeholder_token,
        },
        generate_images: {
            type: "generate_images",
            pipe: { type: "ref", ref: "save_pipeline" },
            prompt: "A <cat-toy> backpack",
            seed: seed,
            generated_image_path: "cat-backpack.png",
        },
    },
}
