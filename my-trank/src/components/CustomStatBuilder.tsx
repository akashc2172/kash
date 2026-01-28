import React, { useState } from 'react';
import { X, Plus } from 'lucide-react';
import './CustomStatBuilder.css';

interface CustomStatBuilderProps {
    availableStats: string[];
    onSave: (customStat: CustomStat) => void;
    onClose: () => void;
    existingStats?: CustomStat[];
}

export interface CustomStat {
    id: string;
    name: string;
    formula: FormulaComponent[];
}

interface FormulaComponent {
    stat: string;
    operation: '+' | '-' | '*' | '/';
    coefficient: number;
}

export const CustomStatBuilder: React.FC<CustomStatBuilderProps> = ({
    availableStats,
    onSave,
    onClose,
    existingStats = []
}) => {
    const [statName, setStatName] = useState('');
    const [components, setComponents] = useState<FormulaComponent[]>([
        { stat: availableStats[0] || '', operation: '+', coefficient: 1 }
    ]);

    const addComponent = () => {
        setComponents([...components, { stat: availableStats[0] || '', operation: '+', coefficient: 1 }]);
    };

    const updateComponent = (index: number, field: keyof FormulaComponent, value: any) => {
        const updated = [...components];
        updated[index] = { ...updated[index], [field]: value };
        setComponents(updated);
    };

    const removeComponent = (index: number) => {
        setComponents(components.filter((_, i) => i !== index));
    };

    const handleSave = () => {
        if (!statName.trim() || components.length === 0) return;

        const customStat: CustomStat = {
            id: `custom_${Date.now()}`,
            name: statName.trim(),
            formula: components
        };

        onSave(customStat);
        onClose();
    };

    const getFormulaPreview = () => {
        return components.map((c, i) => (
            <span key={i}>
                {i > 0 && ` ${c.operation} `}
                {c.coefficient !== 1 && `${c.coefficient} × `}
                {c.stat}
            </span>
        ));
    };

    return (
        <div className="custom-stat-modal-overlay" onClick={onClose}>
            <div className="custom-stat-modal" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>Create Custom Stat</h3>
                    <button className="close-btn" onClick={onClose}>
                        <X size={20} />
                    </button>
                </div>

                <div className="modal-body">
                    <div className="form-group">
                        <label>Stat Name</label>
                        <input
                            type="text"
                            value={statName}
                            onChange={(e) => setStatName(e.target.value)}
                            placeholder="e.g. My Elite Scorer Index"
                            className="stat-name-input"
                        />
                    </div>

                    <div className="formula-builder">
                        <label>Formula Components</label>
                        {components.map((component, index) => (
                            <div key={index} className="formula-component">
                                {index > 0 && (
                                    <select
                                        value={component.operation}
                                        onChange={(e) => updateComponent(index, 'operation', e.target.value)}
                                        className="operation-select"
                                    >
                                        <option value="+">+</option>
                                        <option value="-">-</option>
                                        <option value="*">×</option>
                                        <option value="/">/</option>
                                    </select>
                                )}

                                <input
                                    type="number"
                                    step="0.1"
                                    value={component.coefficient}
                                    onChange={(e) => updateComponent(index, 'coefficient', parseFloat(e.target.value) || 0)}
                                    className="coefficient-input"
                                    placeholder="Weight"
                                />

                                <span className="multiply-symbol">×</span>

                                <select
                                    value={component.stat}
                                    onChange={(e) => updateComponent(index, 'stat', e.target.value)}
                                    className="stat-select"
                                >
                                    {availableStats.map(stat => (
                                        <option key={stat} value={stat}>{stat}</option>
                                    ))}
                                </select>

                                {components.length > 1 && (
                                    <button
                                        onClick={() => removeComponent(index)}
                                        className="remove-component-btn"
                                    >
                                        <X size={16} />
                                    </button>
                                )}
                            </div>
                        ))}

                        <button onClick={addComponent} className="add-component-btn">
                            <Plus size={16} />
                            Add Component
                        </button>
                    </div>

                    <div className="formula-preview">
                        <label>Formula Preview:</label>
                        <div className="preview-text">{getFormulaPreview()}</div>
                    </div>

                    {existingStats.length > 0 && (
                        <div className="existing-stats">
                            <label>Your Custom Stats:</label>
                            <div className="stats-list">
                                {existingStats.map(stat => (
                                    <div key={stat.id} className="stat-item">
                                        <span className="stat-name">{stat.name}</span>
                                        <button className="delete-stat-btn">
                                            <X size={14} />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                <div className="modal-footer">
                    <button onClick={onClose} className="cancel-btn">Cancel</button>
                    <button
                        onClick={handleSave}
                        className="save-btn"
                        disabled={!statName.trim() || components.length === 0}
                    >
                        Save Custom Stat
                    </button>
                </div>
            </div>
        </div>
    );
};
