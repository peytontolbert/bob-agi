# Visual Perception Pipeline

This pipeline tests Bob's visual perception capabilities by running a series of perception tests on the computer screen environment.

## Overview

The pipeline validates:
1. Basic screen capture and perception
2. Element detection and location
3. Visual embedding generation
4. Scene understanding
5. Visual memory and recall
6. Real-time perception stream

## Usage

Run the pipeline with:

```bash
python -m pipelines.visual_perception_pipeline
```

## Test Scenarios

The pipeline runs through several test scenarios:

1. **Basic Perception Test**
   - Captures screen state
   - Validates frame format
   - Tests basic scene understanding

2. **Element Detection Test**  
   - Finds specific UI elements
   - Validates coordinate accuracy
   - Tests element description matching

3. **Visual Memory Test**
   - Generates and stores embeddings
   - Tests similarity search
   - Validates temporal perception

4. **Real-time Processing Test**
   - Tests continuous perception stream
   - Validates frame processing rate
   - Checks embedding generation speed

## Expected Output

The pipeline will output test results including:
- Perception accuracy metrics
- Processing speed measurements  
- Memory retrieval success rate
- Any failed validations

## Error Handling

The pipeline includes robust error handling and will:
- Log all test failures
- Provide detailed error messages
- Clean up test artifacts
- Maintain test isolation