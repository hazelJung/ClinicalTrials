"use client";

import { useState } from "react";
import Shell from "@/components/Shell";
import {
    Container,
    Title,
    Text,
    Card,
    Grid,
    Textarea,
    Slider,
    Button,
    Group,
    Stack,
    Badge,
    Table,
    Progress,
    Alert,
    Box,
    ThemeIcon,
} from "@mantine/core";
import {
    IconFlask,
    IconSearch,
    IconAlertTriangle,
    IconCircleCheck,
    IconAlertCircle,
} from "@tabler/icons-react";

// Mock endpoint results
const mockEndpoints = [
    { name: "NR-AhR", prob: 0.72, positive: true },
    { name: "NR-AR", prob: 0.15, positive: false },
    { name: "NR-AR-LBD", prob: 0.08, positive: false },
    { name: "NR-ER", prob: 0.45, positive: true },
    { name: "NR-ER-LBD", prob: 0.22, positive: false },
    { name: "NR-Aromatase", prob: 0.12, positive: false },
    { name: "NR-PPAR-gamma", prob: 0.09, positive: false },
    { name: "SR-ARE", prob: 0.68, positive: true },
    { name: "SR-ATAD5", prob: 0.31, positive: false },
    { name: "SR-HSE", prob: 0.18, positive: false },
    { name: "SR-MMP", prob: 0.55, positive: true },
    { name: "SR-p53", prob: 0.42, positive: true },
];

const getRiskLevel = (positiveCount: number) => {
    if (positiveCount <= 1) return { level: "LOW", color: "green" };
    if (positiveCount <= 3) return { level: "MEDIUM", color: "yellow" };
    if (positiveCount <= 5) return { level: "HIGH", color: "orange" };
    return { level: "CRITICAL", color: "red" };
};

export default function QSARPredictor() {
    const [loading, setLoading] = useState(false);
    const [smiles, setSmiles] = useState("CC(=O)Oc1ccccc1C(=O)O");
    const [threshold, setThreshold] = useState(20);
    const [results, setResults] = useState<any>(null);

    const handlePredict = async () => {
        setLoading(true);
        try {
            const response = await fetch("http://localhost:8000/api/qsar/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    smiles: smiles,
                    threshold: threshold / 100,
                }),
            });

            if (!response.ok) {
                throw new Error("Prediction failed");
            }

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            setResults(data);
        } catch (error) {
            console.error("Prediction error:", error);
            alert("Failed to run prediction. Ensure backend server is running.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <Shell>
            <Container size="xl">
                <Stack gap="xs" mb="xl">
                    <Title order={2}>QSAR Toxicity Predictor</Title>
                    <Text c="dimmed" size="sm">
                        ClinTox-trained Random Forest Model (Threshold: {threshold / 100})
                    </Text>
                </Stack>

                <Grid gutter="lg">
                    {/* Input Section */}
                    <Grid.Col span={{ base: 12, md: 5 }}>
                        <Card shadow="sm" padding="lg" radius="md" withBorder mb="lg">
                            <Group mb="md">
                                <IconFlask size={20} />
                                <Text fw={600}>Molecule Input</Text>
                            </Group>

                            <Stack gap="md">
                                <Textarea
                                    label="SMILES String"
                                    placeholder="Enter SMILES notation..."
                                    value={smiles}
                                    onChange={(e) => setSmiles(e.target.value)}
                                    minRows={3}
                                />

                                <Box>
                                    <Text size="sm" fw={500} mb={4}>
                                        Decision Threshold: {(threshold / 100).toFixed(2)}
                                    </Text>
                                    <Slider
                                        value={threshold}
                                        onChange={setThreshold}
                                        min={5}
                                        max={50}
                                        step={5}
                                        marks={[
                                            { value: 10, label: "0.10" },
                                            { value: 20, label: "0.20" },
                                            { value: 30, label: "0.30" },
                                            { value: 50, label: "0.50" },
                                        ]}
                                        color="orange"
                                    />
                                    <Text size="xs" c="dimmed" mt="xl">
                                        Lower threshold = Higher recall (more cautious)
                                    </Text>
                                </Box>

                                <Button
                                    fullWidth
                                    size="lg"
                                    leftSection={<IconSearch size={20} />}
                                    onClick={handlePredict}
                                    loading={loading}
                                >
                                    Predict Toxicity
                                </Button>
                            </Stack>
                        </Card>

                        {/* Model Info */}
                        <Card shadow="sm" padding="lg" radius="md" withBorder>
                            <Text fw={600} mb="sm">Model Information</Text>
                            <Table>
                                <Table.Tbody>
                                    <Table.Tr>
                                        <Table.Td c="dimmed">Training Data</Table.Td>
                                        <Table.Td fw={500}>ClinTox (1,484 drugs)</Table.Td>
                                    </Table.Tr>
                                    <Table.Tr>
                                        <Table.Td c="dimmed">Algorithm</Table.Td>
                                        <Table.Td fw={500}>Random Forest + SMOTE</Table.Td>
                                    </Table.Tr>
                                    <Table.Tr>
                                        <Table.Td c="dimmed">Features</Table.Td>
                                        <Table.Td fw={500}>41 RDKit Descriptors</Table.Td>
                                    </Table.Tr>
                                    <Table.Tr>
                                        <Table.Td c="dimmed">Endpoints</Table.Td>
                                        <Table.Td fw={500}>12 Tox21 Assays</Table.Td>
                                    </Table.Tr>
                                </Table.Tbody>
                            </Table>
                        </Card>
                    </Grid.Col>

                    {/* Results Section */}
                    <Grid.Col span={{ base: 12, md: 7 }}>
                        {results ? (
                            <Stack gap="lg">
                                {/* Risk Summary */}
                                <Card shadow="sm" padding="lg" radius="md" withBorder>
                                    <Group justify="space-between" align="flex-start">
                                        <div>
                                            <Text fw={600} mb="xs">Prediction Result</Text>
                                            {results.drug_name && results.drug_name !== "Unknown Compound" && (
                                                <Text size="xl" fw={700} c="blue" mb="xs">
                                                    {results.drug_name}
                                                </Text>
                                            )}
                                            <Text size="sm" c="dimmed" style={{ maxWidth: 400, wordBreak: "break-all" }}>
                                                {results.smiles}
                                            </Text>
                                        </div>
                                        <Badge size="xl" color={results.risk.color} variant="filled">
                                            {results.risk.level} RISK
                                        </Badge>
                                    </Group>

                                    {/* Structure Image */}
                                    {results.structure_image && (
                                        <Box mt="md" style={{ display: "flex", justifyContent: "center" }}>
                                            <div
                                                dangerouslySetInnerHTML={{
                                                    __html: atob(results.structure_image),
                                                }}
                                                style={{
                                                    width: "300px",
                                                    height: "300px",
                                                    border: "1px solid #E9ECEF",
                                                    borderRadius: "8px",
                                                    padding: "10px",
                                                }}
                                            />
                                        </Box>
                                    )}

                                    <Grid mt="lg">
                                        <Grid.Col span={4}>
                                            <Box ta="center">
                                                <ThemeIcon
                                                    size={60}
                                                    radius="xl"
                                                    color={results.risk.color}
                                                    variant="light"
                                                >
                                                    {results.risk.level === "LOW" || results.risk.level === "MEDIUM" ? (
                                                        <IconCircleCheck size={30} />
                                                    ) : (
                                                        <IconAlertTriangle size={30} />
                                                    )}
                                                </ThemeIcon>
                                            </Box>
                                        </Grid.Col>
                                        <Grid.Col span={8}>
                                            <Text size="sm" c="dimmed">Positive Endpoints</Text>
                                            <Text size="xl" fw={700}>
                                                {results.positiveCount} / 12
                                            </Text>
                                            <Progress
                                                value={(results.positiveCount / 12) * 100}
                                                color={results.risk.color}
                                                size="lg"
                                                mt="xs"
                                            />
                                        </Grid.Col>
                                    </Grid>
                                </Card>

                                {/* Endpoint Details */}
                                <Card shadow="sm" padding="lg" radius="md" withBorder>
                                    <Text fw={600} mb="md">Endpoint Breakdown</Text>
                                    <Table>
                                        <Table.Thead>
                                            <Table.Tr>
                                                <Table.Th>Endpoint</Table.Th>
                                                <Table.Th>Probability</Table.Th>
                                                <Table.Th>Result</Table.Th>
                                            </Table.Tr>
                                        </Table.Thead>
                                        <Table.Tbody>
                                            {results.endpoints.map((ep: any) => (
                                                <Table.Tr key={ep.name}>
                                                    <Table.Td fw={500}>{ep.name}</Table.Td>
                                                    <Table.Td>
                                                        <Progress
                                                            value={ep.prob * 100}
                                                            size="lg"
                                                            color={ep.positive ? "red" : "green"}
                                                            style={{ width: 100 }}
                                                        />
                                                    </Table.Td>
                                                    <Table.Td>
                                                        <Badge
                                                            color={ep.positive ? "red" : "green"}
                                                            variant="light"
                                                        >
                                                            {ep.positive ? "POSITIVE" : "NEGATIVE"}
                                                        </Badge>
                                                    </Table.Td>
                                                </Table.Tr>
                                            ))}
                                        </Table.Tbody>
                                    </Table>
                                </Card>
                            </Stack>
                        ) : (
                            <Alert
                                icon={<IconAlertCircle size={24} />}
                                title="No Prediction Results"
                                color="gray"
                                variant="light"
                            >
                                Enter a SMILES string and click "Predict Toxicity" to see results.
                            </Alert>
                        )}
                    </Grid.Col>
                </Grid>
            </Container>
        </Shell>
    );
}
