import numpy as np
import atomnet


def test_math():
    print("=== math ===")
    print("version:", atomnet.math.version())

    x = np.random.randn(4096).astype(np.float32)
    y = np.random.randn(4096).astype(np.float32)

    print("norm:", atomnet.math.norm_np(x))
    print("dot:", atomnet.math.dot_np(x, y))


def test_life():
    print("\n=== life ===")

    x = np.random.randn(1024).astype(np.float32) * 0.02
    life = atomnet.life.PyAtomLife()
    life.observe_np(x, float(np.linalg.norm(x)), 0.7)

    print("life state:", life.state)
    print("life gate:", life.gate())
    print("life stable:", life.is_stable())


def test_hexiron():
    print("\n=== hexiron ===")

    hexiron = atomnet.PyHexiron()

    print("initial state:", hexiron.state())

    hexiron.update(
        1.0,   # energy
        0.8,   # coherence
        0.2,   # entropy
        55.0,  # temperature
    )

    print("after update:", hexiron.state())
    print("stability:", hexiron.stability())
    print("vertices:", hexiron.vertices())
    print("effective lr:", hexiron.effective_lr(0.001))


def test_veron():
    print("\n=== veron ===")

    config = atomnet.PyVeronRuntime.default_config("luna-nano")

    print("model name:", config.model_name)
    print("threads:", config.threads)
    print("layer count:", config.layer_count)

    print(config.summary())

    yaml_text = config.to_yaml()
    print("YAML:")
    print(yaml_text)

    loaded = atomnet.PyVeronRuntime.from_yaml(yaml_text)
    print("loaded model name:", loaded.model_name)


if __name__ == "__main__":
    test_math()
    test_life()
    test_hexiron()
    test_veron()

    print("\nNEW STRUCTURE TEST FINISHED")