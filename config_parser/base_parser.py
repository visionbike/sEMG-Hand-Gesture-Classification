from abc import ABC, abstractmethod
import argparse

__all__ = ['BaseConfigParser']


class BaseConfigParser(ABC):
    """
    The implementation of base config parser.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Base Configuration',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.args = None

    def _print_args(self) -> None:
        """
        Print config arguments.

        :return:
        """

        msg = '---------------- Options ----------------\n'
        cmt = ''
        for k, v in sorted(vars(self.args).items()):
            default = self.parser.get_default(k)
            if v != default:
                cmt = f'\t[default: {str(default)}]'
            msg += f'{str(k):>25} : {str(v):<30} {cmt}\n'
        msg += '----------------   End   ----------------\n'
        print(msg)

    @abstractmethod
    def _add_arguments(self) -> None:
        """
        Add config arguments.
        :return:
        """

        pass

    @abstractmethod
    def init_args(self) -> None:
        """
        Initial config arguments.
        :return:
        """

        pass

    @abstractmethod
    def parse(self) -> None:
        """
        Load configs from the file.
        :return:
        """

        pass
