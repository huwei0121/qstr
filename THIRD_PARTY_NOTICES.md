# Third-party software and provenance

Local Q-STR and experiment adapters use the BSD 3-Clause license in LICENSE.
Bundled upstream code retains its own licenses, copyright statements and source
provenance. Upstream package metadata and attribution are preserved verbatim.

| Component | Bundled location | License |
| --- | --- | --- |
| C³ toolset | third_party/gate/c3_toolset | Apache-2.0; LICENSE and PROVENANCE.json |
| QuOCS | third_party/ising/quocs | Apache-2.0; upstream license and SOURCE.json in parent |
| Clifford decomposition constants from C³ | src/qstr_benchmarks/pulse_support | C3_LICENSE and NOTICE.md |
| ANATRA, MINQ and compatible input checks | src/qstr_benchmarks/qaoa_support/vendor | Individual MIT licenses and SOURCES.json |

Numerical libraries such as NumPy, SciPy, COBYQA, CMA-ES, Nevergrad, Qiskit,
TensorFlow, Py-BOBYQA, SQImFil, IBCDFO and PyNomadBBO are installed through their
package distributions. Their original licenses remain applicable. NOMAD is
provided by its package rather than a copied platform-specific binary.

Task documentation describes which published model, example or reconstruction
each experiment follows. No bundled archive contains private devices, credentials
or paths required to access the original research checkout.
