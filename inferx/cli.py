"""CLI interface for InferX"""

import click
import json
import yaml
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from . import __version__
from .runtime import InferenceEngine
from .utils import FileUtils


@click.group()
@click.version_option(version=__version__, prog_name="InferX")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """InferX - Lightweight ML Inference Runtime
    
    Train how you want. Export how you want. Run with InferX.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def _load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not config_path:
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        click.echo(f"❌ Failed to load config {config_path}: {e}", err=True)
        return {}


def _save_results(results: Dict[str, Any], output_path: Path, format: str = "json") -> None:
    """Save inference results to file"""
    try:
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif format.lower() == "yaml":
            with open(output_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        click.echo(f"✅ Results saved to: {output_path}")
    except Exception as e:
        click.echo(f"❌ Failed to save results: {e}", err=True)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), 
              help="Configuration file path")
@click.option("--output", "-o", type=click.Path(path_type=Path), 
              help="Output file path")
@click.option("--batch-size", type=int, default=1, help="Batch size for inference")
@click.option("--device", type=click.Choice(["auto", "cpu", "gpu"]), default="auto",
              help="Device to run inference on")
@click.option("--runtime", type=click.Choice(["auto", "onnx", "openvino"]), default="auto",
              help="Runtime engine to use")
@click.option("--format", type=click.Choice(["json", "yaml"]), default="json",
              help="Output format")
@click.pass_context
def run(ctx: click.Context, model_path: Path, input_path: Path, 
        config: Optional[Path], output: Optional[Path], batch_size: int,
        device: str, runtime: str, format: str) -> None:
    """Run inference on model with given input"""
    verbose = ctx.obj.get("verbose", False)
    
    try:
        # Load configuration
        user_config = _load_config(config)
        
        # Update config with CLI arguments
        user_config.update({
            "device": device,
            "runtime": runtime
        })
        
        click.echo(f"🚀 Starting inference...")
        click.echo(f"   Model: {model_path}")
        click.echo(f"   Input: {input_path}")
        click.echo(f"   Device: {device}, Runtime: {runtime}")
        
        if config:
            click.echo(f"   Config: {config}")
        
        # Initialize inference engine
        start_time = time.time()
        click.echo("⏳ Loading model...")
        
        engine = InferenceEngine(
            model_path=model_path,
            config=user_config,
            device=device,
            runtime=runtime
        )
        
        load_time = time.time() - start_time
        click.echo(f"✅ Model loaded in {load_time:.3f}s")
        
        # Handle different input types
        if input_path.is_file():
            # Single file inference
            if FileUtils.is_image_file(input_path):
                click.echo("🔍 Running single image inference...")
                inference_start = time.time()
                
                result = engine.predict(input_path)
                
                inference_time = time.time() - inference_start
                click.echo(f"✅ Inference completed in {inference_time:.3f}s")
                
                # Add timing information
                result["timing"] = {
                    "model_load_time": load_time,
                    "inference_time": inference_time,
                    "total_time": load_time + inference_time
                }
                
                # Display basic results
                if verbose:
                    click.echo("\n📊 Results:")
                    click.echo(json.dumps(result, indent=2))
                else:
                    # Show summary
                    click.echo(f"\n📊 Inference Summary:")
                    click.echo(f"   Model type: {result.get('model_type', 'unknown')}")
                    click.echo(f"   Outputs: {result.get('num_outputs', 'unknown')}")
                    click.echo(f"   Inference time: {inference_time:.3f}s")
                
                # Save results if output path specified
                if output:
                    _save_results(result, output, format)
                
            else:
                click.echo(f"❌ Unsupported file type: {input_path.suffix}", err=True)
                return
        
        elif input_path.is_dir():
            # Batch processing
            image_files = FileUtils.get_image_files(input_path)
            
            if not image_files:
                click.echo(f"❌ No image files found in: {input_path}", err=True)
                return
            
            click.echo(f"🔍 Running batch inference on {len(image_files)} images...")
            
            batch_results = []
            total_inference_time = 0
            
            with click.progressbar(image_files, label="Processing images") as bar:
                for image_file in bar:
                    try:
                        inference_start = time.time()
                        result = engine.predict(image_file)
                        inference_time = time.time() - inference_start
                        total_inference_time += inference_time
                        
                        result["file_path"] = str(image_file)
                        result["inference_time"] = inference_time
                        batch_results.append(result)
                        
                    except Exception as e:
                        click.echo(f"❌ Failed to process {image_file}: {e}", err=True)
                        continue
            
            # Create batch summary
            summary = {
                "batch_summary": {
                    "total_images": len(image_files),
                    "successful": len(batch_results),
                    "failed": len(image_files) - len(batch_results),
                    "total_inference_time": total_inference_time,
                    "average_inference_time": total_inference_time / len(batch_results) if batch_results else 0,
                    "model_load_time": load_time
                },
                "results": batch_results
            }
            
            click.echo(f"✅ Batch processing completed!")
            click.echo(f"   Processed: {len(batch_results)}/{len(image_files)} images")
            click.echo(f"   Total time: {total_inference_time:.3f}s")
            click.echo(f"   Average: {total_inference_time/len(batch_results):.3f}s per image" if batch_results else "")
            
            # Save batch results
            if output:
                _save_results(summary, output, format)
            elif not verbose:
                # Show summary for each image
                for result in batch_results[:5]:  # Show first 5
                    file_name = Path(result["file_path"]).name
                    click.echo(f"   {file_name}: {result.get('inference_time', 0):.3f}s")
                if len(batch_results) > 5:
                    click.echo(f"   ... and {len(batch_results) - 5} more")
        
        else:
            click.echo(f"❌ Invalid input path: {input_path}", err=True)
            return
    
    except Exception as e:
        click.echo(f"❌ Inference failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        return


@cli.command()
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.option("--host", default="0.0.0.0", help="Host to bind the server")
@click.option("--port", default=8080, type=int, help="Port to bind the server")
@click.option("--workers", default=1, type=int, help="Number of worker processes")
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path),
              help="Configuration file path")
@click.pass_context
def serve(ctx: click.Context, model_path: Path, host: str, port: int, 
          workers: int, config: Optional[Path]) -> None:
    """Start FastAPI server for model inference"""
    click.echo(f"Starting server for model: {model_path}")
    click.echo(f"Server will run on {host}:{port} with {workers} workers")
    
    if config:
        click.echo(f"Using config: {config}")
    
    # TODO: Implement server startup
    click.echo("🚧 Server implementation coming soon!")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.option("--tag", default="inferx:latest", help="Docker image tag")
@click.option("--optimize", is_flag=True, help="Optimize Docker image size")
@click.option("--compose", is_flag=True, help="Generate docker-compose.yml")
@click.pass_context
def docker(ctx: click.Context, model_path: Path, tag: str, 
           optimize: bool, compose: bool) -> None:
    """Generate Docker container for model deployment"""
    click.echo(f"Generating Docker container for: {model_path}")
    click.echo(f"Image tag: {tag}")
    
    if optimize:
        click.echo("Optimizing for minimal image size")
    
    if compose:
        click.echo("Generating docker-compose.yml")
    
    # TODO: Implement Docker generation
    click.echo("🚧 Docker generation implementation coming soon!")


@cli.command()
@click.option("--template", type=click.Choice(["yolo", "anomalib", "classification"]),
              default="yolo", help="Project template to use")
@click.pass_context
def init(ctx: click.Context, template: str) -> None:
    """Initialize new InferX project with template"""
    click.echo(f"Initializing project with {template} template")
    
    # TODO: Implement project initialization
    click.echo("🚧 Project initialization implementation coming soon!")


def main() -> None:
    """Main entry point for CLI"""
    cli()