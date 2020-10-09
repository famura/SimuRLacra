import pytest
from pytest_lazyfixture import lazy_fixture


class DummyFixture:
    counter = 0

    def __init__(self):
        DummyFixture.counter += 1
        self.count = DummyFixture.counter
        print(f'Hey there I am {self.count}')

    def __del__(self):
        print(f'R.I.P. #{self.count}')


def test_for():
    for i in range(5):
        DummyFixture()


def test_list():
    [DummyFixture() for i in range(5)]


@pytest.fixture(scope='session')
def single_fixed_dummy():
    return DummyFixture()


@pytest.fixture()
def dummy():
    return DummyFixture()


def test_with_fixture(dummy):
    print(dummy)


@pytest.mark.parametrize('index', list(range(4)))
@pytest.mark.parametrize('dummyparam', [lazy_fixture('dummy')]*4)
def test_with_lazy_params(index, dummyparam):
    print(index, dummyparam.count)


@pytest.mark.parametrize('index', list(range(4)))
def test_with_params(index, dummy):
    print(index)


@pytest.mark.parametrize('index', list(range(4)))
def test_with_session_fixture(index, single_fixed_dummy):
    print(index)
