# Visual Perception and Interaction Pipelines

This directory contains test pipelines that validate Bob's visual perception and interaction capabilities across different scenarios.

## Pipeline Overview

The pipelines validate different aspects of the system:

1. **Visual Perception Pipeline**
   - Basic screen capture and perception
   - Element detection and location
   - Visual embedding generation
   - Scene understanding
   - Visual memory and recall
   - Real-time perception stream

2. **Eyesight Screen Pipeline**
   - Screen capture integration
   - Frame buffer management
   - Eyesight processing accuracy
   - Real-time performance metrics

3. **Visual Accuracy Pipeline**
   - Element detection precision
   - Click-point accuracy
   - Multi-turn visual conversation
   - Visual state verification
   - Error recovery strategies

4. **Discord Voice Channel Pipeline**
   - Task-based visual navigation
   - UI element interaction
   - State transition validation
   - Pipeline metrics and monitoring
   - Error handling and recovery

## Usage

Run individual pipelines with:

```bash
# Visual perception tests
python -m pipelines.visual_perception_pipeline

# Screen capture tests
python -m pipelines.eyesight_screen_pipeline

# Accuracy tests
python -m pipelines.visual_accuracy_pipeline

# Task automation tests
python -m pipelines.visual_accuracy_task_pipeline
```

## Pipeline Architecture

Each pipeline follows a common structure:

1. **Initialization**
   - Component setup (vision, eyes, hands)
   - Metrics initialization
   - Debug/logging configuration

2. **Execution Flow**
   - Step-by-step validation
   - Error handling and recovery
   - State verification
   - Performance monitoring

3. **Results & Metrics**
   - Success/failure tracking
   - Timing measurements
   - Accuracy metrics
   - Debug artifacts

## Debug Output

Pipelines generate detailed debug output:
- Test execution logs
- Performance metrics
- Error reports
- Debug visualizations
- State transition data

Debug artifacts are saved to:
```
debug_output/
├── element_detection/
├── screen_captures/
├── click_accuracy/
└── metrics/
```

## Error Handling

The pipelines implement robust error handling:
- Detailed error logging
- Recovery strategies
- Test isolation
- Resource cleanup
- State validation

## Metrics

Key metrics tracked across pipelines:
- Detection accuracy rates
- Processing times
- Success/failure ratios
- Resource utilization
- Error frequencies

## Future Pipeline Development

When creating new pipelines:

1. **Structure**
   - Follow established pipeline architecture
   - Implement proper error handling
   - Include comprehensive metrics
   - Add debug visualization

2. **Integration**
   - Test component interactions
   - Validate state transitions
   - Monitor resource usage
   - Measure performance impact

3. **Documentation**
   - Document pipeline purpose
   - List test scenarios
   - Explain metrics
   - Provide usage examples

## Contributing

When adding new pipelines:
1. Follow existing naming conventions
2. Add comprehensive documentation
3. Include test coverage
4. Implement proper error handling
5. Add relevant metrics