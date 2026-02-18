"""Minimal example of using Academy to implement DeepDriveWE pattern."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from academy.agent import action
from academy.agent import Agent
from academy.agent import loop
from academy.exchange import LocalExchangeFactory
from academy.handle import Handle
from academy.logging import init_logging
from academy.manager import Manager


class SimulationAgent(Agent):
    """Agent for simulation."""

    # A logger for logging exceptions in background tasks.
    __logger: logging.Logger

    def __init__(
        self,
        train_handle: Handle[TrainingAgent],
        inference_handle: Handle[InferenceAgent],
    ) -> None:
        super().__init__()
        self.train_handle = train_handle
        self.inference_handle = inference_handle

    async def agent_on_startup(self) -> None:
        """Startup."""
        self.__logger = logging.getLogger(__class__.__name__)  # type: ignore[name-defined]

        # Log that the agent has started after initializing the logger.
        self.__logger.info('started')

    @action
    async def simulate(self, simulation_id: int) -> None:
        """Simulate a value."""
        result = f'Simulation {simulation_id}'
        self.__logger.info(result)
        # Send the result to the training and inference agents.
        await self.train_handle.receive_simulation_data(result)
        await self.inference_handle.receive_simulation_data(result)


class TrainingAgent(Agent):
    """Agent for training."""

    # A queue for receiving data from the simulation agent.
    __queue: asyncio.Queue[str]

    # A logger for logging exceptions in background tasks.
    __logger: logging.Logger

    def __init__(self, inference_handle: Handle[InferenceAgent]) -> None:
        super().__init__()
        self.inference_handle = inference_handle

    async def agent_on_startup(self) -> None:
        """Startup."""
        self.__logger = logging.getLogger(__class__.__name__)  # type: ignore[name-defined]
        self.__queue = asyncio.Queue()

        # Log that the agent has started after initializing the logger.
        self.__logger.info('started')

    @action
    async def receive_simulation_data(self, data: str) -> None:
        """Receive data."""
        self.__logger.info(f'received simulation data {data}.')
        await self.__queue.put(data)

    @loop
    async def train(self, shutdown: asyncio.Event) -> None:
        """Train a value."""
        while not shutdown.is_set():
            data = await self.__queue.get()
            model_weights_path = f'Trained on data: {data}'
            self.__logger.info(model_weights_path)
            self.__queue.task_done()

            # Send the model weights path to the inference agent.
            await self.inference_handle.receive_model_weights(
                model_weights_path,
            )


class InferenceAgent(Agent):
    """Agent for inference."""

    # An integer to keep track of the iteration
    iteration: int

    # The model weights path
    model_weights_path: str

    # The latest inference result (an incrementing integer for simplicity).
    inference_result: int

    # A queue for receiving data from the simulation agent.
    __queue: asyncio.Queue[str]

    # A lock to ensure that the model weights are not updated while
    # inference is happening.
    __model_lock: asyncio.Lock

    # A logger for logging exceptions in background tasks.
    __logger: logging.Logger

    async def agent_on_startup(self) -> None:
        """Startup."""
        self.__logger = logging.getLogger(__class__.__name__)  # type: ignore[name-defined]
        self.__queue = asyncio.Queue()
        self.__model_lock = asyncio.Lock()

        # Initialize the iteration counter.
        self.iteration = 0

        # Load the initial model weights.
        self.model_weights_path = 'path/to/model/weights.pt'

        # Initialize the inference result.
        self.inference_result = 0

        # Log that the agent has started after initializing the logger.
        self.__logger.info('started')

    @action
    async def receive_simulation_data(self, data: str) -> None:
        """Receive data from the simulation agent."""
        self.__logger.info(f'received simulation data {data}.')
        await self.__queue.put(data)

    @action
    async def receive_model_weights(self, model_weights_path: str) -> None:
        """Receive data from the training agent."""
        self.__logger.info(
            f'received model weights path {model_weights_path}.',
        )
        # Add a lock to ensure that the model weights are not updated
        # while inference is happening.
        async with self.__model_lock:
            self.model_weights_path = model_weights_path

    @action
    async def get_iteration(self) -> int:
        """Get iteration."""
        self.__logger.info(
            f'getting iteration {self.iteration}.',
        )
        return self.iteration

    @action
    async def get_inference_result(self) -> int:
        """Get inference result."""
        self.__logger.info(
            f'getting inference result {self.inference_result}.',
        )
        return self.inference_result

    @loop
    async def infer(self, shutdown: asyncio.Event) -> None:
        """Infer a value."""
        while not shutdown.is_set():
            # Get the simulation data from the queue.
            data = await self.__queue.get()

            # Run inference on the data using the current model weights.
            async with self.__model_lock:
                self.__logger.info(
                    f'inferring on data {data} with '
                    f'model weights {self.model_weights_path}.',
                )
                self.inference_result += 1

            # Update the iteration to signal that inference has been completed
            # on the current data.
            self.iteration += 1
            self.__queue.task_done()


async def main() -> None:
    """Run the main function."""
    init_logging('INFO')

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
    ) as manager:
        inference_handle = await manager.launch(InferenceAgent)
        training_handle = await manager.launch(
            TrainingAgent,
            args=(inference_handle,),
        )
        simulation_handle = await manager.launch(
            SimulationAgent,
            args=(training_handle, inference_handle),
        )

        # Run 2 iterations of the DeepDriveWE pattern.
        for iteration in range(1, 3):
            # Run the first simulation
            await simulation_handle.simulate(iteration)

            # TODO: Can we do better than spin waiting here?
            # Wait for the inference agent to finish an iteration
            while await inference_handle.get_iteration() < iteration:
                await asyncio.sleep(0.1)

            # Get the inference result from the inference agent.
            inference_result = await inference_handle.get_inference_result()
            logging.info(f'Received inference result: {inference_result}')


if __name__ == '__main__':
    asyncio.run(main())
