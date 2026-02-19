"""Minimal example of using Academy to implement DeepDriveWE pattern.

Usage
-----
Run locally (default, no authentication required)::

    python examples/minimal_pattern/main.py

Run via the Academy Exchange Cloud (requires Globus authentication)::

    python examples/minimal_pattern/main.py --exchange globus
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from academy.agent import action
from academy.agent import Agent
from academy.agent import loop
from academy.exchange.cloud.client import HttpExchangeFactory
from academy.exchange.local import LocalExchangeFactory
from academy.handle import Handle
from academy.logging import init_logging
from academy.manager import Manager
from pydantic import BaseModel
from pydantic import Field

EXCHANGE_ADDRESS = 'https://exchange.academy-agents.org'


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

    def __init__(
        self,
        simulation_handles: list[Handle[SimulationAgent]],
    ) -> None:
        super().__init__()
        self.simulation_handles = simulation_handles

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

            # Send the inference results to the simulation agents
            for simulation_handle in self.simulation_handles:
                await simulation_handle.simulate(self.inference_result)

            # Update the iteration to signal that inference has been completed
            # on the current data.
            self.iteration += 1
            self.__queue.task_done()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Minimal DeepDriveWE pattern example.',
    )
    parser.add_argument(
        '--exchange',
        choices=['local', 'globus'],
        default='local',
        help='Exchange type (default: local).',
    )
    return parser.parse_args()


def create_exchange_factory(
    exchange_type: str,
) -> LocalExchangeFactory | HttpExchangeFactory:
    """Create the exchange factory based on the factory type."""
    if exchange_type == 'local':
        return LocalExchangeFactory()

    # NOTE: If using the cloud exchange, run the authentication prior to
    # submitting a batch job script. This will cache a Globus auth session
    # token on the machine that will be reused.

    # Use the HttpExchangeFactory to connect to the Academy Exchange Cloud.
    # This makes all agents talk to each other through the cloud, which
    # allows them to run on different machines with easier setup.
    return HttpExchangeFactory(url=EXCHANGE_ADDRESS, auth_method='globus')


class DeepDriveWeConfig(BaseModel):
    """Configuration for DeepDriveWE pattern."""

    iterations: int = Field(
        default=2,
        description='Number of iterations to run the pattern for.',
    )

    num_simulations: int = Field(
        default=2,
        description='Number of simulation agents to launch.',
    )


async def main() -> None:
    """Run the main function."""
    args = parse_args()
    init_logging('INFO')

    # Load the configuration
    config = DeepDriveWeConfig()

    async with await Manager.from_exchange_factory(
        factory=create_exchange_factory(args.exchange),
        executors=ThreadPoolExecutor(),
    ) as manager:
        # Register the agents with the manager (this will create the
        # mailboxes for the agents).
        reg_inference_agent = await manager.register_agent(InferenceAgent)
        reg_training_agent = await manager.register_agent(TrainingAgent)
        reg_simulation_agents = await asyncio.gather(
            *[
                manager.register_agent(SimulationAgent)
                for _ in range(config.num_simulations)
            ],
        )

        print('num simulation agents registered:', len(reg_simulation_agents))

        # Get the handle of each agent from the manager.
        inference_handle = manager.get_handle(reg_inference_agent)
        training_handle = manager.get_handle(reg_training_agent)
        simulation_handles = [
            manager.get_handle(reg_simulation_agent)
            for reg_simulation_agent in reg_simulation_agents
        ]

        print('num simulation handles:', len(simulation_handles))

        # Launch the agents (this will start the agent_on_startup method of
        # each agent).
        inference_handle = await manager.launch(
            InferenceAgent,
            registration=reg_inference_agent,
            args=(simulation_handles,),
        )

        training_handle = await manager.launch(
            TrainingAgent,
            registration=reg_training_agent,
            args=(inference_handle,),
        )

        # simulation_agents = [
        #     await manager.launch(
        #         SimulationAgent,
        #         args=(training_handle, inference_handle),
        #     )
        #     for _ in range(config.num_simulations)
        # ]

        simulation_agents = await asyncio.gather(
            *[
                manager.launch(
                    SimulationAgent,
                    registration=reg_simulation_agent,
                    args=(training_handle, inference_handle),
                )
                for reg_simulation_agent in reg_simulation_agents
            ],
        )

        # Kick off the first iteration of simulations
        await asyncio.gather(
            *[agent.simulate(1) for agent in simulation_agents],
        )

        # Wait until the inference agent is done
        while await inference_handle.get_iteration() < config.iterations:
            await asyncio.sleep(0.1)

        # Shutdown the agents (this will also shutdown the manager and
        # exchange).
        await asyncio.gather(
            *[
                manager.shutdown(handle, blocking=True)
                for handle in [inference_handle, training_handle]
            ],
        )

        # # Run 2 iterations of the DeepDriveWE pattern.
        # for iteration in range(1, 3):
        #     # Run the first simulation
        #     # await simulation_handle.simulate(iteration)

        #     await asyncio.gather(
        #         *[agent.simulate(iteration) for agent in simulation_agents],
        #     )

        #     # TODO: Can we do better than spin waiting here?
        #     # Wait for the inference agent to finish an iteration
        #     while await inference_handle.get_iteration() < iteration:
        #         await asyncio.sleep(0.1)

        #     # Get the inference result from the inference agent.
        #     inference_result = await inference_handle.get_inference_result()
        #     logging.info(f'Received inference result: {inference_result}')


if __name__ == '__main__':
    asyncio.run(main())
