import { useState } from 'react';
import './App.css'
import { YAMLParser } from './block/yaml-parser';
import { forward, ForwardPair } from './block/forwarder';
import { Block } from './block/interface';

function App() {
  const [yamlString, setYamlString] = useState(`layers:
  - type: conv2d
    params:
      kernel_size: 3
      stride: 1
      padding: 1
  - type: relu
  - type: maxpool2d
    params:
      kernel_size: 2
      stride: 2
      padding: 0
  - type: flatten
  - type: linear`);

  const [inputShape, setInputShape] = useState([1, 32, 32, 3]);
  const [blocks, setBlocks] = useState<Block[]>([]);
  const [result, setResult] = useState<ForwardPair[]>([]);

  return (
    <div className="container">
      <div className="editor-section">
        <h2 style={{textAlign: "center", marginBottom: "1rem"}}>Neural Network Shape Visualizer</h2>
        <div className="input-group">
          <label>Architecture YAML:</label>
          <textarea 
            className="yaml-editor"
            value={yamlString} 
            onChange={(e) => setYamlString(e.target.value)}
            placeholder="Enter your neural network configuration in YAML format..."
          />
        </div>
        
        <div className="input-group">
          <label>Input Shape:</label>
          <input 
            type="text" 
            className="shape-input"
            value={inputShape.toString()} 
            onChange={(e) => setInputShape(e.target.value.split(',').map(Number))}
            placeholder="1,32,32,3"
          />
        </div>

        <div className="button-group">
          <button 
            className="primary-button"
            onClick={() => {
              const blocks = YAMLParser.parseBlocks(yamlString);
              setBlocks(blocks);
            }}
          >
            Compile
          </button>
          <button 
            className="primary-button"
            onClick={() => {
              const result = forward(blocks, inputShape);
              setResult(result);
            }}
          >
            Forward
          </button>
        </div>
      </div>

      <div className="results-section">
        <div className="blocks-display">
          <h2>Network Architecture</h2>
          {blocks.map((block, index) => (
            <div key={index} className="block-item">
              <span className="block-id">{block.id}</span>
            </div>
          ))}
        </div>

        <div className="forward-results">
          <h2>Forward Pass Results</h2>
          {result.map((pair, index) => (
            <div key={index} className="result-item">
              <div className="shape-flow">
                <span className="shape">{pair.inputShape.toString()}</span>
                <span className="arrow">→</span>
                <span className="block-name">{pair.block.id}</span>
                <span className="arrow">→</span>
                <span className={`shape ${!pair.success ? 'error' : ''}`}>
                  {pair.success ? pair.outputShape?.toString() : pair.error}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
      <footer style={{
        textAlign: 'center',
        padding: '1rem',
        marginTop: 'auto',
        borderTop: '1px solid #eee',
        color: '#666'
      }}>
        © {new Date().getFullYear()} KevinZonda Neural Network Shape Visualizer. All rights reserved.
      </footer>
    </div>
  )
}

export default App
