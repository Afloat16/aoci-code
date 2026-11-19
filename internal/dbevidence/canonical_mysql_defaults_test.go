package dbevidence

import (
	"reflect"
	"testing"
)

func TestCanonicalMySQLDefaultValuesPreserveCatalogBytes(t *testing.T) {
	for _, value := range []string{"  active  ", "\t", "first\r\nsecond", "", "active", "CURRENT_TIMESTAMP"} {
		t.Run(value, func(t *testing.T) {
			table := fixtureTable("users")
			table.Engine = EngineMySQL
			table.Columns[0].DefaultExpression = &value
			canonical, data, digest, components, err := CanonicalTable(table)
			if err != nil {
				t.Fatal(err)
			}
			if got := canonical.Columns[0].DefaultExpression; got == nil || *got != value {
				t.Fatalf("catalog default changed during canonicalization: got %v, want %q", got, value)
			}
			var decoded TableEvidence
			if err := decodeStrict(data, &decoded); err != nil {
				t.Fatal(err)
			}
			_, again, againDigest, againComponents, err := CanonicalTable(decoded)
			if err != nil || !reflect.DeepEqual(data, again) || digest != againDigest || components != againComponents {
				t.Fatalf("MySQL default evidence is not canonical after reload: %v", err)
			}
		})
	}
}

func TestCanonicalMySQLDefaultWhitespaceChangesFingerprint(t *testing.T) {
	for _, pair := range [][2]string{{" active ", "active"}, {"first\r\nsecond", "first\nsecond"}, {"\t", ""}} {
		t.Run(pair[0], func(t *testing.T) {
			var digests [2]string
			var components [2]ComponentHashes
			for i, value := range pair {
				table := fixtureTable("users")
				table.Engine = EngineMySQL
				table.Columns[0].DefaultExpression = &value
				_, _, digest, hashes, err := CanonicalTable(table)
				if err != nil {
					t.Fatal(err)
				}
				digests[i], components[i] = digest, hashes
			}
			if digests[0] == digests[1] || !reflect.DeepEqual(changedComponents(components[0], components[1]), []string{"columns"}) {
				t.Fatalf("distinct MySQL default values collapsed: %q and %q", pair[0], pair[1])
			}
		})
	}
}
