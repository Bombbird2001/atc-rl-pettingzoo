import mmap
import platform
import struct
from abc import ABC, abstractmethod
from common.constants import AIRCRAFT_COUNT


os_name = platform.system()
if os_name == "Windows":
    import win32event
elif os_name == "Darwin" or os_name == "Linux":
    import posix_ipc


class GameBridge(ABC):
    # Shared region:
    # 4 bytes constant: [proceed flag(1 byte)] [terminated(1 byte)] [2 bytes padding]
    # 6 bytes per instruction: [action heading(2 bytes)] [action altitude(1 byte)] [action speed(1 byte)] [validity(1 byte)] [1 byte padding]
    # + 52 bytes per aircraft: [reward(4 bytes)] [state(48 bytes (4x chars, 7x floats, 3x ints, 2x byte (for bool), 1x byte (agent ID), 1 byte padding))]
    CONSTANT_FORMAT = "b?xx"
    CONSTANT_SIZE = 4
    PER_INSTRUCTION_FORMAT = "hbbbx"
    PER_INSTRUCTION_SIZE = 6
    PER_AIRCRAFT_FORMAT = "fccccfffffffiiibbbx"
    PER_AIRCRAFT_SIZE = 52
    ADDITIONAL_PADDING_FORMAT = "xx"
    ADDITIONAL_PADDING_SIZE = 2
    FILE_SIZE = CONSTANT_SIZE + AIRCRAFT_COUNT * PER_INSTRUCTION_SIZE + ADDITIONAL_PADDING_SIZE + AIRCRAFT_COUNT * PER_AIRCRAFT_SIZE
    STRUCT_FORMAT = CONSTANT_FORMAT + AIRCRAFT_COUNT * PER_INSTRUCTION_FORMAT + ADDITIONAL_PADDING_FORMAT + AIRCRAFT_COUNT * PER_AIRCRAFT_FORMAT

    @abstractmethod
    def signal_trainer_initialized(self):
        pass

    @abstractmethod
    def signal_reset_sim(self):
        raise NotImplementedError

    @abstractmethod
    def wait_action_ready(self):
        raise NotImplementedError

    @abstractmethod
    def signal_action_done(self):
        raise NotImplementedError

    @abstractmethod
    def signal_reset_after_step(self):
        raise NotImplementedError

    @abstractmethod
    def get_total_state(self):
        raise NotImplementedError

    @abstractmethod
    def get_aircraft_state(self):
        raise NotImplementedError

    @abstractmethod
    def write_actions(self, aircraft_instructions):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @classmethod
    def get_bridge_for_platform(cls, **kwargs):
        if os_name == "Windows":
            return WindowsGameBridge(**kwargs)
        elif os_name == "Darwin" or os_name == "Linux":
            return UnixGameBridge(**kwargs)
        else:
            raise NotImplementedError(f"Unknown OS {os_name}")


class WindowsGameBridge(GameBridge):
    def __init__(self, instance_suffix=""):
        # Create anonymous memory-mapped file with a local name
        self.mm = mmap.mmap(-1, self.__class__.FILE_SIZE, tagname=f"Local\\ATCSharedMem{instance_suffix}")

        # Named events for synchronization
        self.trainer_initialized = win32event.CreateEvent(None, False, False, f"Local\\ATCTrainerInit{instance_suffix}")
        self.reset_sim = win32event.CreateEvent(None, False, False, f"Local\\ATCResetEvent{instance_suffix}")
        self.action_ready = win32event.CreateEvent(None, False, False, f"Local\\ATCActionReadyEvent{instance_suffix}")
        self.action_done = win32event.CreateEvent(None, False, False, f"Local\\ATCActionDoneEvent{instance_suffix}")
        self.reset_after_step = win32event.CreateEvent(None, False, False, f"Local\\ATCResetAfterEvent{instance_suffix}")

    def signal_trainer_initialized(self):
        win32event.SetEvent(self.trainer_initialized)

    def signal_reset_sim(self):
        win32event.SetEvent(self.reset_sim)

    def wait_action_ready(self):
        win32event.WaitForSingleObject(self.action_ready, win32event.INFINITE)

    def signal_action_done(self):
        win32event.SetEvent(self.action_done)

    def signal_reset_after_step(self):
        win32event.SetEvent(self.reset_after_step)

    def get_total_state(self) -> tuple:
        self.mm.seek(0)
        return struct.unpack(self.__class__.STRUCT_FORMAT, self.mm.read(self.__class__.FILE_SIZE))

    def get_aircraft_state(self) -> tuple:
        state_size = self.__class__.PER_AIRCRAFT_SIZE * AIRCRAFT_COUNT
        state_start = self.__class__.FILE_SIZE - state_size
        self.mm.seek(state_start)
        return struct.unpack(
            self.__class__.STRUCT_FORMAT[len(self.__class__.CONSTANT_FORMAT) + AIRCRAFT_COUNT * len(self.__class__.PER_INSTRUCTION_FORMAT) + len(self.__class__.ADDITIONAL_PADDING_FORMAT):],
            self.mm.read(state_size)
        )

    def write_actions(self, aircraft_instructions):
        self.mm.seek(self.__class__.CONSTANT_SIZE)
        self.mm.write(struct.pack(AIRCRAFT_COUNT * self.__class__.PER_INSTRUCTION_FORMAT,*aircraft_instructions))

    def close(self):
        self.mm.close()


class UnixGameBridge(GameBridge):
    def __create_semaphore__(self, name):
        try:
            return posix_ipc.Semaphore(name, posix_ipc.O_CREAT, initial_value=0)
        except posix_ipc.ExistentialError:
            posix_ipc.unlink_semaphore(name)
            return posix_ipc.Semaphore(name, posix_ipc.O_CREAT, initial_value=0)

    def __init__(self, instance_suffix=""):
        try:
            self.shm = posix_ipc.SharedMemory(f"ATCSharedMem{instance_suffix}", posix_ipc.O_CREX, size=self.__class__.FILE_SIZE)
        except posix_ipc.ExistentialError:
            posix_ipc.unlink_shared_memory(f"ATCSharedMem{instance_suffix}")
            self.shm = posix_ipc.SharedMemory(f"ATCSharedMem{instance_suffix}", posix_ipc.O_CREX, size=self.__class__.FILE_SIZE)

        self.mm = mmap.mmap(self.shm.fd, self.shm.size)
        self.shm.close_fd()

        self.trainer_initialized = self.__create_semaphore__(f"ATCTrainerInit{instance_suffix}")
        self.reset_sim = self.__create_semaphore__(f"ATCResetEvent{instance_suffix}")
        self.action_ready = self.__create_semaphore__(f"ATCActionReadyEvent{instance_suffix}")
        self.action_done = self.__create_semaphore__(f"ATCActionDoneEvent{instance_suffix}")
        self.reset_after_step = self.__create_semaphore__(f"ATCResetAfterEvent{instance_suffix}")

    def signal_trainer_initialized(self):
        self.trainer_initialized.release()

    def signal_reset_sim(self):
        self.reset_sim.release()

    def wait_action_ready(self):
        self.action_ready.acquire()

    def signal_action_done(self):
        self.action_done.release()

    def signal_reset_after_step(self):
        self.reset_after_step.release()

    def get_total_state(self) -> tuple:
        self.mm.seek(0)
        return struct.unpack(self.__class__.STRUCT_FORMAT, self.mm.read(self.__class__.FILE_SIZE))

    def get_aircraft_state(self) -> tuple:
        state_size = self.__class__.PER_AIRCRAFT_SIZE * AIRCRAFT_COUNT
        state_start = self.__class__.FILE_SIZE - state_size
        self.mm.seek(state_start)
        return struct.unpack(
            self.__class__.STRUCT_FORMAT[len(self.__class__.CONSTANT_FORMAT) + AIRCRAFT_COUNT * len(self.__class__.PER_INSTRUCTION_FORMAT) + len(self.__class__.ADDITIONAL_PADDING_FORMAT):],
            self.mm.read(state_size)
        )

    def write_actions(self, aircraft_instructions):
        self.mm.seek(self.__class__.CONSTANT_SIZE)
        self.mm.write(struct.pack(AIRCRAFT_COUNT * self.__class__.PER_INSTRUCTION_FORMAT,*aircraft_instructions))

    def close(self):
        self.mm.close()
        self.shm.unlink()
        self.reset_sim.unlink()
        self.action_ready.unlink()
        self.action_done.unlink()
        self.reset_after_step.unlink()
