from .base import BaseVisualizer
from core.logger import get_logger

logger = get_logger()


class GradioVisualizer(BaseVisualizer):
    def __init__(self, config):
        super().__init__(config)
        self.interface = None

    def setup(self):
        try:
            import gradio as gr
            self.interface = gr.Interface(
                fn=self._predict,
                inputs="text",
                outputs="text",
                title="Model Evaluation"
            )
            return True
        except ImportError:
            return False

    def visualize(self, data):
        if not self.interface:
            success = self.setup()
            if not success:
                logger.warning("Gradio is not installed. Visualization will be skipped.")
                return
        # Launch the interface with sample data
        self.interface.launch(share=self.config.get('share', False))
    
    def save(self):
        # Gradio interfaces are not saved to files
        pass
    
    def _predict(self, text):
        # Sample prediction function
        return f"Processed: {text}"

class WandbVisualizer(BaseVisualizer):
    def __init__(self, config):
        super().__init__(config)
        self.run = None
    
    def setup(self):
        try:
            import wandb
            self.run = wandb.init(
                project=self.config.get('project', 'model_eval'),
                name=self.config.get('name', 'eval_run')
            )
            return True
        except ImportError:
            raise ImportError("Please install wandb: pip install wandb")
    
    def visualize(self, data):
        if not self.run:
            self.setup()
        # Log metrics to Wandb
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    self.run.log(value, step=0)
                else:
                    self.run.log({key: value}, step=0)
    
    def save(self):
        if self.run:
            self.run.finish()

class SwanLabVisualizer(BaseVisualizer):
    def __init__(self, config):
        super().__init__(config)
    
    def setup(self):
        try:
            import swanlab
            swanlab.init(
                project=self.config.get('project', 'model_eval'),
                experiment_name=self.config.get('name', 'eval_run')
            )
            return True
        except ImportError:
            raise ImportError("Please install swanlab: pip install swanlab")
    
    def visualize(self, data):
        import swanlab
        # Log metrics to SwanLab
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        swanlab.log({f"{key}_{sub_key}": sub_value})
                else:
                    swanlab.log({key: value})
    
    def save(self):
        # SwanLab automatically saves runs
        pass

class ClearMLVisualizer(BaseVisualizer):
    def __init__(self, config):
        super().__init__(config)
        self.task = None
    
    def setup(self):
        try:
            from clearml import Task
            self.task = Task.init(
                project_name=self.config.get('project', 'model_eval'),
                task_name=self.config.get('name', 'eval_run')
            )
            return True
        except ImportError:
            raise ImportError("Please install clearml: pip install clearml")
    
    def visualize(self, data):
        if not self.task:
            self.setup()
        # Log metrics to ClearML
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        self.task.get_logger().report_scalar(
                            title=key, 
                            series=sub_key, 
                            value=sub_value, 
                            iteration=0
                        )
                else:
                    self.task.get_logger().report_scalar(
                        title="Metrics", 
                        series=key, 
                        value=value, 
                        iteration=0
                    )
    
    def save(self):
        if self.task:
            self.task.close()
