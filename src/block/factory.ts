import { LeakyReLU, ReLU, Sigmoid, Tanh } from "./activation";
import { Block, Linear, Conv2d, MaxPool2d, Flatten, Reshape, TransposeConv2d, BatchNorm2d, Dropout } from "./interface";

export function factoryBlock(name : string, params : Record<string, number | number[]>) : Block {
    switch (name) {
        case "linear":
            return new Linear();
        case "conv2d":
            return new Conv2d(params["kernel_size"] as number, params["stride"] as number, params["padding"] as number);
        case "maxpool2d":
            return new MaxPool2d(params["kernel_size"] as number, params["stride"] as number, params["padding"] as number);
        case "flatten":
            return new Flatten();
        case "transposeconv2d":
            return new TransposeConv2d(params["kernel_size"] as number, params["stride"] as number, params["padding"] as number);
        case "reshape":
            return new Reshape(params["shape"] as number[]);
        case "dropout":
            if (params["p"] === undefined) {
                return new Dropout();
            }
            return new Dropout(params["p"] as number);
        case "batchnorm2d":
            return new BatchNorm2d(params["shape"] as number);
        case "relu":
            return new ReLU();
        case "tanh":
            return new Tanh();
        case "leakyrelu":
            return new LeakyReLU();
        case "sigmoid":
            return new Sigmoid();
        default:
            throw new Error(`Unknown block type: ${name}`);
    }
}