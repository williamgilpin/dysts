import random
import unittest

import numpy as np

import dysts.flows as dfl
import dysts.maps as dmp
from dysts.base import DynMap, DynSys, DynSysDelay
from dysts.systems import get_attractor_list

NUM_TEST_SYSTEMS = 10


class TestDynamicalSystems(unittest.TestCase):
    def test_random_continuous_systems(self):
        continuous_systems = get_attractor_list(sys_class="continuous_no_delay")
        random_systems = random.sample(
            continuous_systems, min(NUM_TEST_SYSTEMS, len(continuous_systems))
        )

        for system_name in random_systems:
            with self.subTest(system=system_name):
                print(f"Testing {system_name}")
                system = getattr(dfl, system_name)()
                self.assertIsInstance(system, DynSys)

                sol = system.make_trajectory(1000, return_times=True)
                self.assertIsInstance(sol, tuple)
                self.assertEqual(len(sol), 2)
                self.assertIsInstance(sol[0], np.ndarray)
                self.assertIsInstance(sol[1], np.ndarray)
                self.assertEqual(sol[0].shape[0], 1000)
                self.assertEqual(sol[1].shape[0], 1000)

    def test_random_delay_systems(self):
        delay_systems = get_attractor_list(sys_class="delay")
        random_systems = random.sample(
            delay_systems, min(NUM_TEST_SYSTEMS, len(delay_systems))
        )

        for system_name in random_systems:
            with self.subTest(system=system_name):
                print(f"Testing {system_name}")
                system = getattr(dfl, system_name)()
                self.assertIsInstance(system, DynSysDelay)

                sol = system.make_trajectory(1000, return_times=True)
                self.assertIsInstance(sol, tuple)
                self.assertEqual(len(sol), 2)
                self.assertIsInstance(sol[0], np.ndarray)
                self.assertIsInstance(sol[1], np.ndarray)
                self.assertEqual(sol[0].shape[0], 1000)
                self.assertEqual(sol[1].shape[0], 1000)

    def test_random_discrete_maps(self):
        discrete_maps = get_attractor_list(sys_class="discrete")
        random_systems = random.sample(
            discrete_maps, min(NUM_TEST_SYSTEMS, len(discrete_maps))
        )

        for system_name in random_systems:
            with self.subTest(system=system_name):
                print(f"Testing {system_name}")
                system = getattr(dmp, system_name)()
                self.assertIsInstance(system, DynMap)

                sol = system.make_trajectory(1000)
                self.assertIsInstance(sol, np.ndarray)
                self.assertEqual(sol.shape[0], 1000)


if __name__ == "__main__":
    unittest.main()
