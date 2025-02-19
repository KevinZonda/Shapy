import { Block } from "./interface";

export interface ForwardPair {
    block : Block;
    inputShape : number[];
    outputShape : number[] | undefined;
    success : boolean;
    error : string | undefined;
}

export function forward(blocks : Block[], inputShape : number[]) {
    const outShapes : ForwardPair[] = [];
    for (const block of blocks) {
        const pair : ForwardPair = {
            block : block,
            inputShape : inputShape,
            outputShape : undefined,
            success : false,
            error : undefined
        }
        try {
            pair.outputShape = block.forward(inputShape);
            pair.success = true;
        } catch (error) {
            if (error instanceof Error) {
                pair.error = error.message;
            } else {
                pair.error = String(error).toString();
            }
        }
        outShapes.push(pair);
    }
    return outShapes;
}