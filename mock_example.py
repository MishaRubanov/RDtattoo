from typing import Protocol
from unittest import mock


class DragonInterface(Protocol):
    def breathfire(self) -> str: ...

    def fly(self) -> str: ...


class FireDragon(DragonInterface):
    def breathfire(self) -> str:
        return "Intense flames!"

    def fly(self) -> str:
        return "Flying with fire wings!"


class IceDragon(DragonInterface):
    def breathfire(self) -> str:
        return "Freezing breath!"

    def fly(self) -> str:
        return "Soaring through snow clouds!"


class MechanicalDragon(DragonInterface):
    def breathfire(self) -> str:
        return "Shooting flames from fuel tank!"

    def fly(self) -> str:
        return "Engine-powered flight!"


# Mocking the protocol
print("\nMock Dragon:")
mock_dragon = mock.Mock(spec=DragonInterface)
mock_dragon.fly.return_value = "Soaring!"
mock_dragon.breathfire.return_value = "Mock fire!"

print("\nMock dragon calls:")
print(mock_dragon.mock_calls)
