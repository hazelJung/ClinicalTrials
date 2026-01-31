"use client";

import { useState } from "react";
import Shell from "@/components/Shell";
import {
    Container,
    Title,
    Text,
    Card,
    Grid,
    TextInput,
    NumberInput,
    Select,
    MultiSelect,
    Slider,
    Button,
    Group,
    Stack,
    Divider,
    Badge,
    Table,
    Progress,
    Box,
    Alert,
} from "@mantine/core";
import {
    IconPlayerPlay,
    IconFlask,
    IconUsers,
    IconActivity,
    IconAlertCircle,
} from "@tabler/icons-react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    BarChart,
    Bar,
} from "recharts";

// Mock PK data
const mockPKData = [
    { time: 0, concentration: 0, TO: 0 },
    { time: 1, concentration: 850, TO: 45 },
    { time: 7, concentration: 620, TO: 78 },
    { time: 14, concentration: 450, TO: 92 },
    { time: 21, concentration: 380, TO: 94 },
    { time: 28, concentration: 320, TO: 95 },
    { time: 42, concentration: 210, TO: 88 },
    { time: 56, concentration: 140, TO: 72 },
];

const ethnicityOptions = [
    { value: "EUR", label: "European" },
    { value: "EAS", label: "East Asian" },
    { value: "AFR", label: "African" },
    { value: "AMR", label: "Admixed American" },
    { value: "SAS", label: "South Asian" },
];

export default function MPBPKSimulator() {
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<any>(null);

    // Drug parameters
    const [drugName, setDrugName] = useState("Custom Antibody");
    const [kd, setKd] = useState<number | string>(1.0);
    const [dose, setDose] = useState<number | string>(10);
    const [mw, setMw] = useState<number | string>(150);
    const [charge, setCharge] = useState<string | null>("0");

    // Target parameters
    const [t0, setT0] = useState<number | string>(10);
    const [halflife, setHalflife] = useState<number | string>(200);

    // Cohort configuration
    const [ethnicities, setEthnicities] = useState<string[]>(["EUR", "EAS"]);
    const [populationSize, setPopulationSize] = useState<number | string>(100);
    const [maleRatio, setMaleRatio] = useState(50);
    const [weightMean, setWeightMean] = useState<number | string>(70);
    const [weightSd, setWeightSd] = useState<number | string>(12);

    const handleSimulate = async () => {
        setLoading(true);
        try {
            const response = await fetch("http://localhost:8000/api/mpbpk/simulate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    drug_name: drugName,
                    kd: Number(kd),
                    dose: Number(dose),
                    mw: Number(mw),
                    charge: Number(charge),
                    t0: Number(t0),
                    halflife: Number(halflife),
                    ethnicities: ethnicities,
                    population_size: Number(populationSize),
                    male_ratio: maleRatio,
                    weight_mean: Number(weightMean),
                    weight_sd: Number(weightSd),
                }),
            });

            if (!response.ok) {
                throw new Error("Simulation failed");
            }

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            setResults(data);
        } catch (error: any) {
            console.error("Simulation error:", error);
            alert(`Simulation failed: ${error.message}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Shell>
            <Container size="xl">
                <Stack gap="xs" mb="xl">
                    <Title order={2}>mPBPK Simulator</Title>
                    <Text c="dimmed" size="sm">
                        Minimal PBPK Model for Monoclonal Antibody Pharmacokinetics
                    </Text>
                </Stack>

                <Grid gutter="lg">
                    {/* Left Column - Input Parameters */}
                    <Grid.Col span={{ base: 12, md: 5 }}>
                        {/* Drug Parameters */}
                        <Card shadow="sm" padding="lg" radius="md" withBorder mb="lg">
                            <Group mb="md">
                                <IconFlask size={20} />
                                <Text fw={600}>Drug Parameters</Text>
                            </Group>

                            <Stack gap="sm">
                                <TextInput
                                    label="Drug Name"
                                    value={drugName}
                                    onChange={(e) => setDrugName(e.target.value)}
                                />
                                <Grid>
                                    <Grid.Col span={6}>
                                        <NumberInput
                                            label="KD (nM)"
                                            value={kd}
                                            onChange={setKd}
                                            min={0.01}
                                            max={100}
                                            decimalScale={2}
                                        />
                                    </Grid.Col>
                                    <Grid.Col span={6}>
                                        <NumberInput
                                            label="Dose (mg/kg)"
                                            value={dose}
                                            onChange={setDose}
                                            min={0.1}
                                            max={50}
                                            decimalScale={1}
                                        />
                                    </Grid.Col>
                                </Grid>
                                <Grid>
                                    <Grid.Col span={6}>
                                        <NumberInput
                                            label="MW (kDa)"
                                            value={mw}
                                            onChange={setMw}
                                            min={100}
                                            max={200}
                                        />
                                    </Grid.Col>
                                    <Grid.Col span={6}>
                                        <Select
                                            label="Charge"
                                            value={charge}
                                            onChange={setCharge}
                                            data={[
                                                { value: "-5", label: "-5 (Negative)" },
                                                { value: "0", label: "0 (Neutral)" },
                                                { value: "5", label: "+5 (Positive)" },
                                            ]}
                                        />
                                    </Grid.Col>
                                </Grid>
                            </Stack>
                        </Card>

                        {/* Target Parameters */}
                        <Card shadow="sm" padding="lg" radius="md" withBorder mb="lg">
                            <Group mb="md">
                                <IconActivity size={20} />
                                <Text fw={600}>Target Parameters</Text>
                            </Group>

                            <Grid>
                                <Grid.Col span={6}>
                                    <NumberInput
                                        label="T0 Baseline (nM)"
                                        value={t0}
                                        onChange={setT0}
                                        min={1}
                                        max={100}
                                    />
                                </Grid.Col>
                                <Grid.Col span={6}>
                                    <NumberInput
                                        label="Half-life (hr)"
                                        value={halflife}
                                        onChange={setHalflife}
                                        min={10}
                                        max={1000}
                                    />
                                </Grid.Col>
                            </Grid>
                        </Card>

                        {/* Cohort Configuration */}
                        <Card shadow="sm" padding="lg" radius="md" withBorder mb="lg">
                            <Group mb="md">
                                <IconUsers size={20} />
                                <Text fw={600}>Cohort Configuration</Text>
                            </Group>

                            <Stack gap="sm">
                                <MultiSelect
                                    label="Ethnicities"
                                    value={ethnicities}
                                    onChange={setEthnicities}
                                    data={ethnicityOptions}
                                    placeholder="Select populations"
                                />
                                <NumberInput
                                    label="Population Size (N)"
                                    value={populationSize}
                                    onChange={setPopulationSize}
                                    min={10}
                                    max={1000}
                                    step={10}
                                />
                                <Box>
                                    <Text size="sm" fw={500} mb={4}>
                                        Gender Ratio: {maleRatio}% Male / {100 - maleRatio}% Female
                                    </Text>
                                    <Slider
                                        value={maleRatio}
                                        onChange={setMaleRatio}
                                        marks={[
                                            { value: 0, label: "All F" },
                                            { value: 50, label: "50/50" },
                                            { value: 100, label: "All M" },
                                        ]}
                                    />
                                </Box>
                                <Grid mt="sm">
                                    <Grid.Col span={6}>
                                        <NumberInput
                                            label="Weight Mean (kg)"
                                            value={weightMean}
                                            onChange={setWeightMean}
                                            min={40}
                                            max={120}
                                        />
                                    </Grid.Col>
                                    <Grid.Col span={6}>
                                        <NumberInput
                                            label="Weight SD (kg)"
                                            value={weightSd}
                                            onChange={setWeightSd}
                                            min={5}
                                            max={30}
                                        />
                                    </Grid.Col>
                                </Grid>
                            </Stack>
                        </Card>

                        <Button
                            fullWidth
                            size="lg"
                            leftSection={<IconPlayerPlay size={20} />}
                            onClick={handleSimulate}
                            loading={loading}
                        >
                            Run Simulation
                        </Button>
                    </Grid.Col>

                    {/* Right Column - Results */}
                    <Grid.Col span={{ base: 12, md: 7 }}>
                        {results ? (
                            <Stack gap="lg">
                                {/* Summary Card */}
                                <Card shadow="sm" padding="lg" radius="md" withBorder>
                                    <Group justify="space-between" mb="md">
                                        <Text fw={600}>Simulation Results</Text>
                                        <Badge
                                            color={results.summary.successRate > 80 ? "green" : "yellow"}
                                            size="lg"
                                        >
                                            {Number(results.summary.successRate).toFixed(1)}% Success
                                        </Badge>
                                    </Group>

                                    <Grid>
                                        <Grid.Col span={3}>
                                            <Text size="xs" c="dimmed">TO Trough</Text>
                                            <Text size="lg" fw={700}>{Number(results.summary.toTrough).toFixed(2)}%</Text>
                                        </Grid.Col>
                                        <Grid.Col span={3}>
                                            <Text size="xs" c="dimmed">Cmax</Text>
                                            <Text size="lg" fw={700}>{Number(results.summary.cMax).toFixed(0)} nM</Text>
                                        </Grid.Col>
                                        <Grid.Col span={3}>
                                            <Text size="xs" c="dimmed">AUC</Text>
                                            <Text size="lg" fw={700}>{Number(results.summary.auc).toFixed(0)}</Text>
                                        </Grid.Col>
                                        <Grid.Col span={3}>
                                            <Text size="xs" c="dimmed">Toxicity Risk</Text>
                                            <Text size="lg" fw={700} c={results.summary.toxicityRisk > 0 ? "red" : "green"}>{Number(results.summary.toxicityRisk).toFixed(1)}%</Text>
                                        </Grid.Col>
                                    </Grid>
                                </Card>

                                {/* PK Profile Chart */}
                                <Card shadow="sm" padding="lg" radius="md" withBorder>
                                    <Text fw={600} mb="md">PK Profile (Concentration & TO%)</Text>
                                    <ResponsiveContainer width="100%" height={300}>
                                        <LineChart data={results.pkData} margin={{ top: 5, right: 20, left: 10, bottom: 25 }}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis
                                                dataKey="time"
                                                label={{ value: 'Time (Days)', position: 'insideBottom', offset: -5 }}
                                                height={50}
                                            />
                                            <YAxis
                                                yAxisId="left"
                                                label={{ value: 'Conc (nM)', angle: -90, position: 'insideLeft', offset: 10 }}
                                            />
                                            <YAxis
                                                yAxisId="right"
                                                orientation="right"
                                                domain={[0, 100]}
                                                label={{ value: 'TO (%)', angle: 90, position: 'insideRight', offset: 10 }}
                                            />
                                            <Tooltip formatter={(value: any) => Number(value).toFixed(1)} />
                                            <Legend verticalAlign="top" height={36} />
                                            <Line yAxisId="left" type="monotone" dataKey="concentration" stroke="#228BE6" strokeWidth={2} name="Concentration" />
                                            <Line yAxisId="right" type="monotone" dataKey="TO" stroke="#40C057" strokeWidth={2} name="Target Occupancy" />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </Card>

                                {/* Cohort Breakdown */}
                                <Card shadow="sm" padding="lg" radius="md" withBorder>
                                    <Text fw={600} mb="md">Cohort Breakdown</Text>
                                    <Table>
                                        <Table.Thead>
                                            <Table.Tr>
                                                <Table.Th>Ethnicity</Table.Th>
                                                <Table.Th>N</Table.Th>
                                                <Table.Th>Mean TO%</Table.Th>
                                                <Table.Th>Pass Rate</Table.Th>
                                            </Table.Tr>
                                        </Table.Thead>
                                        <Table.Tbody>
                                            {results.cohortBreakdown.map((row: any) => (
                                                <Table.Tr key={row.ethnicity}>
                                                    <Table.Td fw={500}>{row.ethnicity}</Table.Td>
                                                    <Table.Td>{row.n}</Table.Td>
                                                    <Table.Td>{Number(row.toMean).toFixed(2)}%</Table.Td>
                                                    <Table.Td>
                                                        <Progress value={row.passPct} size="lg" color="green" />
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
                                title="No Simulation Results"
                                color="gray"
                                variant="light"
                            >
                                Configure the parameters on the left and click "Run Simulation" to see results.
                            </Alert>
                        )}
                    </Grid.Col>
                </Grid>
            </Container>
        </Shell>
    );
}
