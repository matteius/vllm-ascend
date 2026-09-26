# SPDX-License-Identifier: Apache-2.0
"""Print the exact synthetic repository context used in the 23K-token runs."""

MODULE_COUNT = 180
BASE_CAPACITY = 32
CAPACITY_VARIANTS = 17

TEMPLATE = """# module worker_{index}.py
class Worker{index}:
    def __init__(self, capacity={capacity}):
        self.capacity = capacity
        self.items = []
        self.closed = False

    def submit(self, item):
        if self.closed:
            raise RuntimeError(closed)
        if len(self.items) >= self.capacity:
            raise OverflowError(full)
        self.items.append(item)

    def take(self):
        if not self.items:
            return None
        return self.items.pop(0)

    def close(self):
        self.closed = True
"""


def main() -> None:
    # This is input text for review, including the original undefined names.
    print("The following repository excerpt is context for a programming discussion.\n")
    print(
        "\n".join(
            TEMPLATE.format(index=index, capacity=BASE_CAPACITY + index % CAPACITY_VARIANTS)
            for index in range(MODULE_COUNT)
        ),
        end="",
    )


if __name__ == "__main__":
    main()
