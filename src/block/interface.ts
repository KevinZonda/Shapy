export interface Block {
    id: string;
    forward(shape : number[]): number[];
}

export class Linear implements Block {
    id : string = "linear";
    forward(shape : number[]): number[] {
        if (shape.length !== 2) {
            throw new Error("Linear block must have exactly two inputs (batch_size, *)");
        }
        return shape;
    }
}

export class Conv2d implements Block {
    id : string = "conv2d";
    kernel_size : number;
    stride : number;
    padding : number;

    constructor(kernel_size : number, stride : number = 1, padding : number = 0) {
        this.kernel_size = kernel_size;
        this.stride = stride;
        this.padding = padding;
    }
    
    forward(shape : number[]): number[] {
        if (shape.length !== 4) {
            throw new Error("Conv2d block must have exactly four inputs (batch_size, channels, height, width)");
        }
        const [batch_size, channels, height, width] = shape;
        
        const output_height = Math.floor((height + 2 * this.padding - this.kernel_size) / this.stride + 1);
        const output_width = Math.floor((width + 2 * this.padding - this.kernel_size) / this.stride + 1);
        
        if (output_height <= 0 || output_width <= 0) {
            throw new Error("Invalid output dimensions. Check kernel size, stride and padding values");
        }
        
        return [batch_size, channels, output_height, output_width];
    }
}

export class MaxPool2d implements Block {
    id : string = "maxpool2d";
    kernel_size : number;
    stride : number;
    padding : number;

    constructor(kernel_size : number, stride : number | null = null, padding : number = 0) {
        this.kernel_size = kernel_size;
        this.stride = stride ?? kernel_size;
        this.padding = padding;
    }
    
    forward(shape : number[]): number[] {
        if (shape.length !== 4) {
            throw new Error("MaxPool2d block must have exactly four inputs (batch_size, channels, height, width)");
        }
        const [batch_size, channels, height, width] = shape;
        
        const output_height = Math.floor((height + 2 * this.padding - this.kernel_size) / this.stride + 1);
        const output_width = Math.floor((width + 2 * this.padding - this.kernel_size) / this.stride + 1);
        
        if (output_height <= 0 || output_width <= 0) {
            throw new Error("Invalid output dimensions. Check kernel size, stride and padding values");
        }
        
        return [batch_size, channels, output_height, output_width];
    }
}

export class Flatten implements Block {
    id : string = "flatten";
    forward(shape : number[]): number[] {
        if (shape.length < 2) {
            throw new Error("Flatten block must have at least two dimensions (batch_size, *)");
        }
        let sum = 1;
        for (let i = 1; i < shape.length; i++) {
            sum *= shape[i];
        }
        return [shape[0], sum];
    }
}

export class Dropout implements Block {
    id : string = "dropout";
    p: number;

    constructor(p: number = 0.5) {
        if (p < 0 || p > 1) {
            throw new Error("Dropout probability must be between 0 and 1");
        }
        this.p = p;
    }

    forward(shape: number[]): number[] {
        return shape;
    }
}

export class BatchNorm2d implements Block {
    id : string = "batchnorm2d";
    num_features: number;
    
    constructor(num_features: number) {
        this.num_features = num_features;
    }
    
    forward(shape: number[]): number[] {
        if (shape.length !== 4) {
            throw new Error("BatchNorm2d block must have exactly four inputs (batch_size, channels, height, width)");
        }
        const [batch_size, channels, height, width] = shape;
        if (channels !== this.num_features) {
            throw new Error(`Expected ${this.num_features} channels but got ${channels}`);
        }
        return [batch_size, channels, height, width];
    }
}

export class TransposeConv2d implements Block {
    id : string = "transposeconv2d";
    kernel_size: number;
    stride: number;
    padding: number;
    
    constructor(kernel_size: number, stride: number = 1, padding: number = 0) {
        this.kernel_size = kernel_size;
        this.stride = stride;
        this.padding = padding;
    }

    forward(shape: number[]): number[] {
        if (shape.length !== 4) {
            throw new Error("TransposeConv2d block must have exactly four inputs (batch_size, channels, height, width)");
        }

        const [batch_size, channels, height, width] = shape;
        
        const output_height = (height - 1) * this.stride - 2 * this.padding + this.kernel_size;
        const output_width = (width - 1) * this.stride - 2 * this.padding + this.kernel_size;
        
        if (output_height <= 0 || output_width <= 0) {
            throw new Error("Invalid output dimensions. Check kernel size, stride and padding values");
        }
        
        return [batch_size, channels, output_height, output_width];
    }
}

export class Reshape implements Block {
    id : string = "reshape";
    shape: number[];

    constructor(shape: number[]) {
        this.shape = shape;
    }

    forward(shape: number[]): number[] {
        // Calculate total elements in input shape
        const inputElements = shape.reduce((a, b) => a * b, 1);
        
        // Calculate total elements in target shape
        const targetElements = this.shape.reduce((a, b) => a * b, 1);
        
        // Verify shapes are compatible
        if (inputElements !== targetElements) {
            throw new Error(`Cannot reshape tensor of size ${shape} (${inputElements} elements) into shape ${this.shape} (${targetElements} elements)`);
        }
        
        return this.shape;
    }
}

export class Unflatten implements Block {
    id: string = "unflatten";
    target_shape: number[];

    constructor(target_shape: number[]) {
        this.target_shape = target_shape;
    }

    forward(shape: number[]): number[] {
        if (shape.length !== 2) {
            throw new Error("Unflatten block expects input shape of [batch_size, flattened_dim]");
        }

        const [batch_size, flattened_dim] = shape;
        
        // Calculate total elements in target shape
        const targetElements = this.target_shape.reduce((a, b) => a * b, 1);
        
        // Verify shapes are compatible
        if (flattened_dim !== targetElements) {
            throw new Error(`Cannot unflatten tensor of size [${batch_size}, ${flattened_dim}] into shape [${batch_size}, ${this.target_shape.join(', ')}]. Flattened dimension ${flattened_dim} does not match product of target dimensions ${targetElements}`);
        }
        
        return [batch_size, ...this.target_shape];
    }
}

