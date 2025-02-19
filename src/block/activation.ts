import { Block } from "./interface";

export class Activation implements Block {
    id : string = "activation";
    forward(shape : number[]): number[] {
        return shape;
    }
}

export class ReLU extends Activation {
    override id : string = "relu";
}

export class Tanh extends Activation {
    override id : string = "tanh";
}

export class LeakyReLU extends Activation {
    override id : string = "leakyrelu";
}

export class Sigmoid extends Activation {
    override id : string = "sigmoid";
}
