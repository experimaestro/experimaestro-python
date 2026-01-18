"""Tests for mock_modules functionality.

These tests verify that the module mocking system properly handles:
- Decorator patterns (@torch.no_grad, @torch.compile, etc.)
- Class inheritance (torch.autograd.Function with .apply method)
- Generic type subscripting (nn.Module[T])
- Nested attribute access
"""

import sys
import pytest


@pytest.fixture
def clean_torch_modules():
    """Remove any torch-related modules from sys.modules before/after tests."""
    # Store original modules
    torch_modules = {k: v for k, v in sys.modules.items() if k.startswith("torch")}
    # Remove them
    for k in torch_modules:
        del sys.modules[k]

    yield

    # Cleanup after test - remove fake modules
    fake_modules = [k for k in sys.modules.keys() if k.startswith("torch")]
    for k in fake_modules:
        del sys.modules[k]


@pytest.fixture
def mock_torch(clean_torch_modules):
    """Set up torch mocking for tests.

    Note: No decorators list needed - all mocked objects automatically work as decorators.
    """
    from experimaestro.experiments import mock_modules

    finder = mock_modules(
        ["torch", "lightning_fabric", "torchmetrics", "pytorch_lightning"]
    )

    yield finder

    # Remove finder from meta_path
    if finder in sys.meta_path:
        sys.meta_path.remove(finder)


class TestAutomaticDecorators:
    """Test that all mocked objects automatically work as decorators."""

    def test_decorator_without_parens(self, mock_torch):
        """Test @decorator pattern (without parentheses)."""
        import torch

        @torch.no_grad
        def my_function():
            return 42

        assert my_function() == 42

    def test_decorator_with_parens(self, mock_torch):
        """Test @decorator() pattern (with parentheses, no args)."""
        import torch

        @torch.no_grad()
        def my_function():
            return 42

        assert my_function() == 42

    def test_decorator_with_args(self, mock_torch):
        """Test @decorator(arg=value) pattern."""
        import torch

        @torch.compile(mode="reduce-overhead")
        def my_function(x):
            return x + 1

        assert my_function(41) == 42

    def test_nested_decorator_without_parens(self, mock_torch):
        """Test @module.submodule.decorator pattern."""
        import torch

        @torch.jit.script
        def my_function(x):
            return x

        assert my_function("hello") == "hello"

    def test_nested_decorator_with_parens(self, mock_torch):
        """Test @module.submodule.decorator() pattern."""
        import torch

        @torch.cuda.amp.autocast()
        def my_function():
            return "autocast"

        assert my_function() == "autocast"

    def test_nested_decorator_with_args(self, mock_torch):
        """Test @module.submodule.decorator(arg=value) pattern."""
        import torch

        @torch.jit.script(optimize=True)
        def my_function():
            return "optimized"

        assert my_function() == "optimized"

    def test_stacked_decorators(self, mock_torch):
        """Test multiple stacked decorators."""
        import torch

        @torch.no_grad
        @torch.compile
        @torch.jit.script
        def my_function():
            return "stacked"

        assert my_function() == "stacked"

    def test_stacked_decorators_with_mixed_patterns(self, mock_torch):
        """Test stacked decorators with mixed patterns."""
        import torch

        @torch.inference_mode()
        @torch.compile(mode="max-autotune")
        @torch.jit.export
        def my_function():
            return "mixed"

        assert my_function() == "mixed"

    def test_class_decorator(self, mock_torch):
        """Test decorator on a class."""
        import torch

        @torch.compile
        class MyModel:
            def forward(self):
                return "compiled"

        model = MyModel()
        assert model.forward() == "compiled"

    def test_class_decorator_with_args(self, mock_torch):
        """Test decorator with args on a class."""
        import torch

        @torch.compile(fullgraph=True)
        class MyModel:
            def forward(self):
                return "fullgraph"

        model = MyModel()
        assert model.forward() == "fullgraph"


class TestDeprecationWarning:
    """Test that using decorators parameter raises deprecation warning."""

    def test_decorators_param_warns(self, clean_torch_modules):
        """Test that passing decorators parameter raises DeprecationWarning."""
        import warnings
        from experimaestro.experiments import mock_modules

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            finder = mock_modules(
                ["torch"],
                decorators=["torch.no_grad"],
            )
            # Clean up
            if finder in sys.meta_path:
                sys.meta_path.remove(finder)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "decorators" in str(w[0].message)

    def test_no_warning_without_decorators(self, clean_torch_modules):
        """Test that no warning is raised without decorators parameter."""
        import warnings
        from experimaestro.experiments import mock_modules

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            finder = mock_modules(["torch"])
            # Clean up
            if finder in sys.meta_path:
                sys.meta_path.remove(finder)

            # Filter for DeprecationWarning only
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0


class TestNoopDecorators:
    """Test that decorators work correctly."""

    def test_no_grad_decorator_without_parens(self, mock_torch):
        """Test @torch.no_grad without parentheses."""
        import torch

        @torch.no_grad
        def my_function():
            return 42

        assert my_function() == 42

    def test_no_grad_decorator_with_parens(self, mock_torch):
        """Test @torch.no_grad() with parentheses."""
        import torch

        @torch.no_grad()
        def my_function():
            return 42

        assert my_function() == 42

    def test_inference_mode_decorator(self, mock_torch):
        """Test @torch.inference_mode() decorator."""
        import torch

        @torch.inference_mode()
        def my_function():
            return "inference"

        assert my_function() == "inference"

    def test_compile_decorator(self, mock_torch):
        """Test @torch.compile decorator."""
        import torch

        @torch.compile
        def my_function(x):
            return x * 2

        assert my_function(21) == 42

    def test_compile_decorator_with_args(self, mock_torch):
        """Test @torch.compile(mode='reduce-overhead') decorator."""
        import torch

        @torch.compile(mode="reduce-overhead")
        def my_function(x):
            return x + 1

        assert my_function(41) == 42

    def test_jit_script_decorator(self, mock_torch):
        """Test @torch.jit.script decorator."""
        import torch

        @torch.jit.script
        def my_function(x):
            return x

        assert my_function("hello") == "hello"

    def test_jit_unused_decorator(self, mock_torch):
        """Test @torch.jit.unused decorator."""
        import torch

        @torch.jit.unused
        def helper_function():
            return "unused"

        assert helper_function() == "unused"

    def test_jit_export_decorator(self, mock_torch):
        """Test @torch.jit.export decorator."""
        import torch

        @torch.jit.export
        def exported_function():
            return "exported"

        assert exported_function() == "exported"

    def test_jit_ignore_decorator(self, mock_torch):
        """Test @torch.jit.ignore decorator."""
        import torch

        @torch.jit.ignore
        def ignored_function():
            return "ignored"

        assert ignored_function() == "ignored"


class TestAutogradFunction:
    """Test torch.autograd.Function behavior."""

    def test_autograd_function_apply_exists(self, mock_torch):
        """Test that subclasses of torch.autograd.Function have .apply method."""
        import torch

        class MyFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * 2

        # The key test: .apply should exist and be callable
        assert hasattr(MyFunction, "apply")
        assert callable(MyFunction.apply)

    def test_autograd_function_apply_callable(self, mock_torch):
        """Test that .apply can be called."""
        import torch

        class BCEWithLogLoss(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

        # This is the pattern that was failing
        bce_with_logits_loss = BCEWithLogLoss.apply
        assert callable(bce_with_logits_loss)

        # Should be able to call it (result doesn't matter, just that it doesn't raise)
        bce_with_logits_loss(1, 2, 3)

    def test_autograd_function_nested(self, mock_torch):
        """Test multiple autograd.Function classes."""
        import torch

        class Func1(torch.autograd.Function):
            pass

        class Func2(torch.autograd.Function):
            pass

        assert hasattr(Func1, "apply")
        assert hasattr(Func2, "apply")
        assert Func1.apply is not Func2.apply or callable(Func1.apply)


class TestNNModule:
    """Test torch.nn.Module behavior."""

    def test_nn_module_inheritance(self, mock_torch):
        """Test that nn.Module can be inherited from."""
        import torch

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = 42

            def forward(self, x):
                return x * self.value

        model = MyModel()
        assert model.value == 42

    def test_nn_module_with_layers(self, mock_torch):
        """Test that nn module layers can be used."""
        import torch

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        # Should not raise
        model = MyModel()
        assert hasattr(model, "linear")


class TestGenericTypes:
    """Test generic type subscripting."""

    def test_tensor_subscript(self, mock_torch):
        """Test Tensor[...] subscripting."""
        import torch

        # Should not raise
        TensorType = torch.Tensor[int]
        assert TensorType is not None

    def test_module_subscript(self, mock_torch):
        """Test Module[T] subscripting."""
        import torch

        ModuleType = torch.nn.Module[str]
        assert ModuleType is not None


class TestNestedAttributes:
    """Test deeply nested attribute access."""

    def test_deeply_nested_attribute(self, mock_torch):
        """Test accessing deeply nested attributes."""
        import torch

        # Should not raise
        _ = torch.cuda.amp.autocast
        _ = torch.distributed.rpc.functions

    def test_submodule_import(self, mock_torch):
        """Test importing submodules."""
        import torch.nn.functional as F

        # Should not raise
        assert F is not None

    def test_nested_class_inheritance(self, mock_torch):
        """Test inheriting from nested classes."""
        import torch

        class MyCallback(torch.utils.data.Dataset):
            pass

        # Should not raise
        _ = MyCallback()


class TestMixedUsage:
    """Test realistic mixed usage patterns."""

    def test_typical_model_definition(self, mock_torch):
        """Test a typical PyTorch model definition."""
        import torch

        @torch.compile
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)

            @torch.jit.export
            def forward(self, x):
                return self.linear(x)

            @torch.jit.ignore
            def debug_info(self):
                return "debug"

        model = Model()
        assert model.debug_info() == "debug"

    def test_autograd_function_with_decorators(self, mock_torch):
        """Test autograd.Function with decorators."""
        import torch

        class MyOp(torch.autograd.Function):
            @staticmethod
            @torch.jit.unused
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad):
                return grad

        # Key assertion
        assert hasattr(MyOp, "apply")
        MyOp.apply(1)

    def test_inference_function(self, mock_torch):
        """Test inference function pattern."""
        import torch

        @torch.inference_mode()
        @torch.compile
        def run_inference(model, x):
            return model(x)

        # Should be callable
        assert callable(run_inference)


class TestImportPatterns:
    """Test different import patterns for mocked modules."""

    def test_import_torch_nn_as_nn(self, mock_torch):
        """Test: import torch.nn as nn => class A(nn.Module)"""
        import torch.nn as nn

        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.value = 42

        model = MyModel()
        assert model.value == 42

    def test_import_torch_dot_nn_module(self, mock_torch):
        """Test: import torch => class A(torch.nn.Module)"""
        import torch

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = 42

        model = MyModel()
        assert model.value == 42

    def test_from_torch_nn_import_module(self, mock_torch):
        """Test: from torch.nn import Module => class A(Module)"""
        from torch.nn import Module

        class MyModel(Module):
            def __init__(self):
                super().__init__()
                self.value = 42

        model = MyModel()
        assert model.value == 42

    def test_autograd_import_torch_then_access(self, mock_torch):
        """Test: import torch => class A(torch.autograd.Function)"""
        import torch

        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

        assert hasattr(MyFunc, "apply")
        assert callable(MyFunc.apply)

    def test_autograd_import_as_alias(self, mock_torch):
        """Test: import torch.autograd as autograd => class A(autograd.Function)"""
        import torch.autograd as autograd

        class MyFunc(autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

        assert hasattr(MyFunc, "apply")
        assert callable(MyFunc.apply)

    def test_autograd_from_import(self, mock_torch):
        """Test: from torch.autograd import Function => class A(Function)"""
        from torch.autograd import Function

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, x):
                return x

        assert hasattr(MyFunc, "apply")
        assert callable(MyFunc.apply)

    def test_nested_import_pattern(self, mock_torch):
        """Test: from torch.nn.utils import rnn"""
        from torch.nn.utils import rnn

        # Should be able to access attributes
        _ = rnn.pack_padded_sequence
        assert rnn is not None


class TestMultipleInheritance:
    """Test inheriting from both real and mocked classes."""

    def test_inherit_from_real_and_mocked_class(self, mock_torch):
        """Test: class Foo(RealClass, MockedClass) - no metaclass conflict."""
        import torch

        # Create a real class (not mocked)
        class RealScorer:
            def score(self):
                return 42

        # Inherit from both a real class and a mocked class
        class MyScorer(RealScorer, torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = 1

        scorer = MyScorer()
        assert scorer.score() == 42
        assert scorer.value == 1

    def test_inherit_from_real_class_with_metaclass(self, mock_torch):
        """Test inheriting from a class that has a custom metaclass."""
        import torch

        # A metaclass for the real class
        class RealMeta(type):
            pass

        class RealBase(metaclass=RealMeta):
            pass

        # Should not raise metaclass conflict
        class Combined(RealBase, torch.nn.Module):
            pass

        obj = Combined()
        assert obj is not None

    def test_multiple_mocked_bases(self, mock_torch):
        """Test inheriting from multiple mocked classes."""
        import torch
        import pytorch_lightning as pl

        class MyModel(torch.nn.Module, pl.LightningModule):
            def __init__(self):
                super().__init__()

        model = MyModel()
        assert model is not None

    def test_mixed_inheritance_with_apply(self, mock_torch):
        """Test that .apply still works with mixed inheritance."""
        import torch

        class RealBase:
            pass

        class MyFunction(RealBase, torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

        # .apply should still be accessible
        assert hasattr(MyFunction, "apply")
        assert callable(MyFunction.apply)


class TestOtherMockedModules:
    """Test other mocked modules work correctly."""

    def test_pytorch_lightning(self, mock_torch):
        """Test pytorch_lightning module."""
        import pytorch_lightning as pl

        class MyModule(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.value = 1

        module = MyModule()
        assert module.value == 1

    def test_lightning_fabric(self, mock_torch):
        """Test lightning_fabric module."""
        import lightning_fabric

        # Should not raise
        _ = lightning_fabric.Fabric

    def test_torchmetrics(self, mock_torch):
        """Test torchmetrics module."""
        import torchmetrics

        class MyMetric(torchmetrics.Metric):
            pass

        # Should not raise
        _ = MyMetric()
