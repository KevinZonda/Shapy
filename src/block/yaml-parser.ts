import { Block } from "./interface";
import { factoryBlock } from "./factory";
import yaml from "js-yaml";

interface BlockConfig {
    type: string;
    params?: Record<string, number | number[]>;
}

interface ModelConfig {
    layers: BlockConfig[];
}

export class YAMLParser {
    static parseBlocks(yamlString: string): Block[] {
        try {
            const config = yaml.load(yamlString) as ModelConfig;
            
            if (!config.layers || !Array.isArray(config.layers)) {
                throw new Error("YAML must contain a 'layers' array");
            }

            return config.layers.map(layer => {
                if (!layer.type) {
                    throw new Error("Each layer must have a 'type' field");
                }
                return factoryBlock(layer.type, layer.params || {});
            });
        } catch (error) {
            if (error instanceof Error) {
                throw new Error(`Failed to parse YAML: ${error.message}`);
            }
            throw error;
        }
    }
}